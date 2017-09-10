import gzip
import numpy as np
import os
from tqdm import tqdm

from .util import download, get_dataset_dir


_DATASET_NAME = 'covertype'
_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/' + \
       'covtype/covtype.data.gz'
_PROCESSED_SUBDIR = 'processed'
_NPY_BASENAME = 'data.npy'


def _process(gz_filename, processed_dir, verbose):
    os.mkdir(processed_dir)
    lines = gzip.open(gz_filename).readlines()
    if verbose:
        lines = tqdm(lines, leave=False)
    arrs = []
    for line in lines:
        nn = list(map(int, line.decode('utf-8').split(',')))
        arr = np.array(nn, dtype='int32')
        arrs.append(arr)
    arr = np.stack(arrs)
    npy_filename = os.path.join(processed_dir, _NPY_BASENAME)
    np.save(npy_filename, arr)


def _load(processed_dir, val_frac, verbose):
    npy_filename = os.path.join(processed_dir, _NPY_BASENAME)
    arr = np.load(npy_filename)
    np.random.shuffle(arr)
    split = int(len(arr) * val_frac)
    x_train = arr[split:, :-1]
    y_train = arr[split:, -1] - 1
    x_val = arr[:split, :-1]
    y_val = arr[:split, -1] - 1
    return (x_train, y_train), (x_val, y_val)


def load_cover_type(val_frac=0.2, verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    processed_dir = os.path.join(dataset_dir, _PROCESSED_SUBDIR)
    if not os.path.exists(processed_dir):
        gz_filename = os.path.join(dataset_dir, os.path.basename(_URL))
        download(_URL, gz_filename, verbose)
        _process(gz_filename, processed_dir, verbose)
    return _load(processed_dir, val_frac, verbose)
