import time
import torch
import torch.nn as nn
import numpy as np
from torch.optim.optimizer import Optimizer
import math


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
      self.val = 0
      self.avg = 0
      self.sum = 0
      self.count = 0

  def update(self, val, n=1):
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count


def obtain_accuracy(output, target, topk=(1,)):
  with torch.no_grad():
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # bs*k
    pred = pred.t()  # t: transpose, k*bs
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 1*bs --> k*bs

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


def obtain_per_class_accuracy(predictions, xtargets):
  top1 = torch.argmax(predictions, dim=1)
  cls2accs = []
  for cls in sorted(list(set(xtargets.tolist()))):
    selects  = xtargets == cls
    accuracy = (top1[selects] == xtargets[selects]).float().mean() * 100
    cls2accs.append( accuracy.item() )
  return sum(cls2accs) / len(cls2accs)


def convert_secs2time(epoch_time, string=True):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  if string:
    need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
    return need_time
  else:
    return need_hour, need_mins, need_secs

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d-%X'
  string = '[{:}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def time_for_file():
  ISOTIMEFORMAT='%d-%h-at-%H-%M-%S'
  string = '{:}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def spm_to_tensor(sparse_mx):
  sparse_mx = sparse_mx.tocoo().astype(np.float32)
  indices = torch.from_numpy(np.vstack(
          (sparse_mx.row, sparse_mx.col))).long()
  values = torch.from_numpy(sparse_mx.data)
  shape = torch.Size(sparse_mx.shape)
  return torch.sparse.FloatTensor(indices, values, shape)
