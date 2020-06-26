#!/usr/bin/env python3
import os, sys, time, json, random, argparse
import collections
import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / 'fast-exps' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

from data.meta_dataset_reader import MetaDatasetEpisodeReader, TRAIN_METADATASET_NAMES, ALL_METADATASET_NAMES
from models.new_model_helpers import get_extractors, extract_features
# from xmodels.new_model_helpers import get_extractors, extract_features
from models.models_dict import DATASET_MODELS_DICT
from utils import convert_secs2time, time_string, AverageMeter
from paths import META_RECORDS_ROOT

from config_utils import Logger


def load_config():

  parser = argparse.ArgumentParser(description='Train prototypical networks')
  parser.add_argument('--save_dir', type=str, help="The saved path in dir.")

  # model args
  parser.add_argument('--model.backbone', default='resnet18', help="Use ResNet18 for experiments (default: False)")
  parser.add_argument('--model.classifier', type=str, default='cosine', choices=['none', 'linear', 'cosine'], help="Do classification using cosine similatity between activations and weights")

  # train args
  parser.add_argument('--train.max_iter', type=int, default=10000, help='number of epochs to train (default: 10000)')
  parser.add_argument('--eval.max_iter', type=int, default=600, help='number of epochs to train (default: 10000)')

  xargs = vars(parser.parse_args())
  return xargs


def extract_eval_dataset(backbone, mode, extractors, all_test_datasets, test_loader, num_iters, logger, save_dir):
  # dataset_models = DATASET_MODELS_DICT[backbone]

  logger.print('\n{:} starting extract the {:} mode by {:} iters.'.format(time_string(), mode, save_dir, num_iters))
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.compat.v1.Session(config=config) as session:
    for idata, test_dataset in enumerate(all_test_datasets):
      logger.print('===>>> {:} --->>> {:02d}/{:02d} --->>> {:}'.format(time_string(), idata, len(all_test_datasets), test_dataset))
      x_save_dir = save_dir / '{:}-{:}'.format(mode, num_iters) / '{:}'.format(test_dataset)
      x_save_dir.mkdir(parents=True, exist_ok=True)
      for idx in tqdm(range(num_iters)):
        # extract image features and labels
        if mode == "val":
          sample = test_loader.get_validation_task(session, test_dataset)
        elif mode == "test":
          sample = test_loader.get_test_task(session, test_dataset)
        else:
          raise ValueError("invalid mode:{}".format(mode))

        with torch.no_grad():
          context_labels = sample['context_labels']
          target_labels  = sample['target_labels']
          # batch x #extractors x #features
          context_features = extract_features(extractors, sample['context_images'])
          target_features = extract_features(extractors, sample['target_images'])
          to_save_info = {'context_features': context_features.cpu(),
                          'context_labels': context_labels.cpu(),
                          'target_features': target_features.cpu(),
                          'target_labels': target_labels.cpu()}
          save_name = x_save_dir / '{:06d}.pth'.format(idx)
          torch.save(to_save_info, save_name)


def main(xargs):

  # set up logger
  log_dir = Path(xargs['save_dir']).resolve()
  log_dir.mkdir(parents=True, exist_ok=True)
  #log_dir = "./NEWsave/{}_{}_allcache_{}_{}_{}_{}_{}_{}".format(args['train.optimizer'], args['train.scheduler'], args['prop.n_hop'], args['prop.temp'], args['prop.nonlinear'], args['prop.transform'], args['prop.layer_type'], args['prop.layer_type.att_space'])

  logger = Logger(str(log_dir), 888)
  logger.print('{:} --- args ---'.format(time_string()))
  for key, value in xargs.items():
    logger.print('  [{:10s}] : {:}'.format(key, value))
  logger.print('{:} --- args ---'.format(time_string()))

  # Setting up datasets
  extractor_domains = TRAIN_METADATASET_NAMES
  all_val_datasets  = TRAIN_METADATASET_NAMES
  all_test_datasets = ALL_METADATASET_NAMES
  train_loader_lst  = [MetaDatasetEpisodeReader('train', [d], [d], all_test_datasets) for d in extractor_domains]
  val_loader        = MetaDatasetEpisodeReader('val' , extractor_domains, extractor_domains, all_test_datasets)
  test_loader       = MetaDatasetEpisodeReader('test', extractor_domains, extractor_domains, all_test_datasets)
  class_name_dict   = collections.OrderedDict()
  for d in extractor_domains:
    with open("{:}/{:}/dataset_spec.json".format(META_RECORDS_ROOT, d)) as f:
      data = json.load(f) 
      class_name_dict[d] = data['class_names']

  # initialize the feature extractors
  dataset_models = DATASET_MODELS_DICT[xargs['model.backbone']]
  extractors = get_extractors(extractor_domains, dataset_models, xargs['model.backbone'], xargs['model.classifier'], False)

  extract_eval_dataset(xargs['model.backbone'], 'test', extractors, all_test_datasets, test_loader, xargs['eval.max_iter'], logger, log_dir)
  # stop at here
  extract_eval_dataset(xargs['model.backbone'], 'val' , extractors, all_val_datasets , val_loader , xargs['eval.max_iter'], logger, log_dir)

  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True

  xsave_dir = log_dir / 'train-{:}'.format(xargs['train.max_iter'])
  xsave_dir.mkdir(parents=True, exist_ok=True)
  logger.print('{:} save into {:}'.format(time_string(), xsave_dir))
  with tf.compat.v1.Session(config=config) as session:
    for idx in tqdm(range(xargs['train.max_iter'])):
      if random.random() > 0.5:
        ep_domain  = extractor_domains[0]
      else:
        ep_domain  = random.choice(extractor_domains[1:])
      domain_idx   = extractor_domains.index(ep_domain)
      train_loader = train_loader_lst[domain_idx]
      samples      = train_loader.get_train_task(session)
      # import pdb; pdb.set_trace()
      domain_extractor = extractors[ep_domain]

      with torch.no_grad():
        # batch x #extractors x #features
        context_labels   = samples['context_labels'].cpu()
        target_labels    = samples['target_labels'].cpu()
        context_features = extract_features(extractors, samples['context_images'])
        target_features  = extract_features(extractors, samples['target_images'])
        to_save_info = {'context_features': context_features.cpu(),
                        'context_labels': context_labels.cpu(),
                        'target_features': target_features.cpu(),
                        'target_labels': target_labels.cpu(),
                        'ep_domain': ep_domain,
                        'domain_idx': domain_idx}
        save_name = xsave_dir / '{:06d}.pth'.format(idx)
        torch.save(to_save_info, save_name)


if __name__ == '__main__':
  xargs = load_config()
  main(xargs)
