import gzip
import numpy as np
import os
from tqdm import tqdm

from .util import download, get_dataset_dir


_DATASET_NAME = 'higgs_boson'
_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/' + \
       '00280/HIGGS.csv.gz'
_PROCESSED_SUBDIR = 'processed'
_NPY_BASENAME = 'data.npy'
_SPLIT = -500000


def _process(gz_filename, processed_dir, verbose):
    os.mkdir(processed_dir)
    in_ = gzip.open(gz_filename)
    if verbose == 2:
        in_ = tqdm(in_, total=11000000, leave=False)
    arrs = []
    for line in in_:
        ss = line.decode('utf-8').split(',')
        ff = list(map(float, ss))
        arr = np.array(ff, dtype='float32')
        arrs.append(arr)
    in_.close()
    arr = np.stack(arrs)
    npy_filename = os.path.join(processed_dir, _NPY_BASENAME)
    np.save(npy_filename, arr)


def _load(processed_dir, verbose):
    npy_filename = os.path.join(processed_dir, _NPY_BASENAME)
    arr = np.load(npy_filename)
    x_train = arr[:_SPLIT, 1:]
    x_val = arr[_SPLIT:, 1:]
    y_train = arr[:_SPLIT, 0]
    y_val = arr[_SPLIT:, 0]
    return (x_train, y_train), (x_val, y_val)


def load_higgs_boson(verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    processed_dir = os.path.join(dataset_dir, _PROCESSED_SUBDIR)
    if not os.path.exists(processed_dir):
        gz_filename = os.path.join(dataset_dir, os.path.basename(_URL))
        download(_URL, gz_filename, verbose)
        _process(gz_filename, processed_dir, verbose)
    return _load(processed_dir, verbose)
