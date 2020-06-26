from collections import OrderedDict
from .EpisodeMetadata import EpisodeMetadata


def get_eval_datasets(root, dataset_names, num=600):
  #eval_dataset_names = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower']
  datasets = OrderedDict()
  for name in dataset_names:
    dataset = EpisodeMetadata(root, name, num)
    datasets[name] = dataset
  return datasets


def get_train_dataset(root, num=10000):
  return EpisodeMetadata(root, 'train-{:}'.format(num), num)
