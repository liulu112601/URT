import os, sys
from pathlib import Path

PROJECT_ROOT = str((Path(__file__).parent / '..').resolve())
META_DATASET_ROOT = os.environ['META_DATASET_ROOT']
META_RECORDS_ROOT = os.environ['RECORDS']
META_DATA_ROOT = '/'.join(META_RECORDS_ROOT.split('/')[:-1])
