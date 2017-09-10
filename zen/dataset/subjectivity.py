import numpy as np
import os
from random import shuffle
import tarfile

from .util import download, get_dataset_dir


_DATASET_NAME = 'subjectivity'
_URL = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/' + \
       'rotten_imdb.tar.gz'
_PROCESSED_SUBDIR = 'processed'
_OBJ_TGZ_PATH = 'plot.tok.gt9.5000'
_OBJ_BASENAME = 'objective.txt'
_SBJ_TGZ_PATH = 'quote.tok.gt9.5000'
_SBJ_BASENAME = 'subjective.txt'


def _extract(tar, tar_path, processed_dir, processed_basename, verbose):
    data = tar.extractfile(tar_path).read()
    text = data.decode('latin-1')
    out_filename = os.path.join(processed_dir, processed_basename)
    out = open(out_filename, 'wb')
    for line in text.strip().split('\n'):
        line = (line + '\n').encode('utf-8')
        out.write(line)
    out.close()


def _process(tgz_filename, processed_dir, verbose):
    os.mkdir(processed_dir)
    tar = tarfile.open(tgz_filename, 'r:gz')
    _extract(tar, _OBJ_TGZ_PATH, processed_dir, _OBJ_BASENAME, verbose)
    _extract(tar, _SBJ_TGZ_PATH, processed_dir, _SBJ_BASENAME, verbose)


def _read(processed_dir, basename, verbose):
    filename = os.path.join(processed_dir, basename)
    texts = []
    for line in open(filename):
        text = line[:-1]
        texts.append(text)
    return texts


def _blend(obj, sbj, val_frac):
    samples = list(map(lambda text: (text, 0), obj)) + \
              list(map(lambda text: (text, 1), sbj))
    shuffle(samples)
    x, y = list(zip(*samples))
    y = np.array(y, dtype='int64')
    split = int(len(x) * val_frac)
    x_train = x[split:]
    x_val = x[:split]
    y_train = y[split:]
    y_val = y[:split]
    return (x_train, y_train), (x_val, y_val)


def _load(processed_dir, val_frac, verbose):
    obj = _read(processed_dir, _OBJ_BASENAME, verbose)
    sbj = _read(processed_dir, _SBJ_BASENAME, verbose)
    return _blend(obj, sbj, val_frac)


def load_subjectivity(val_frac=0.2, verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    processed_dir = os.path.join(dataset_dir, _PROCESSED_SUBDIR)
    if not os.path.exists(processed_dir):
        tgz_filename = os.path.join(dataset_dir, os.path.basename(_URL))
        download(_URL, tgz_filename, verbose)
        _process(tgz_filename, processed_dir, verbose)
    return _load(processed_dir, val_frac, verbose)
