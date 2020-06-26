import os, sys
from pathlib import Path
import numpy as np
import scipy.misc
import pprint

pp = pprint.PrettyPrinter(indent=4)

try:
  from StringIO import StringIO  # Python 2.7
except ImportError:
  from io import BytesIO  # Python 3.x

from .utils import time_for_file

class Logger(object):

  def __init__(self, log_dir, seed):
    """Create a summary writer logging to log_dir."""
    self.log_dir = Path("{:}".format(str(log_dir)))
    if not self.log_dir.exists(): os.makedirs(str(self.log_dir))

    self.log_file = '{:}/log-{:}-date-{:}.txt'.format(self.log_dir, seed, time_for_file())
    self.file_writer = open(self.log_file, 'w')

  def checkpoint(self, name):
    return self.log_dir / name

  def print(self, string, fprint=True, is_pp=False):
    if is_pp: pp.pprint (string)
    else:     print(string)
    if fprint:
      self.file_writer.write('{:}\n'.format(string))
      self.file_writer.flush()

  def close(self):
    self.file_writer.close()
