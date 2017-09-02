from glob import glob
import numpy as np
import os
from random import shuffle
import tarfile
from time import time
from tqdm import tqdm

from .util import download, get_dataset_dir


_DATASET_NAME = 'imdb'
_URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
_TGZ_SUBDIR = 'aclImdb'
_PROCESSED_SUBDIR = 'processed'


def _extract(dataset_dir, tgz_basename, verbose):
    local = os.path.join(dataset_dir, tgz_basename)
    if verbose:
        print('Extracting...')
        t0 = time()
    tar = tarfile.open(local, 'r:gz')
    tar.extractall(dataset_dir)
    if verbose:
        t = time() - t0
        print('...took %.3f sec.' % t)


def _combine(pos, neg):
    pos = list(map(lambda s: (s, 1), pos))
    neg = list(map(lambda s: (s, 0), neg))
    data = pos + neg
    shuffle(data)
    return data


def _process_split_polarity(tgz_dir, processed_dir, split, polarity, verbose):
    if verbose:
        print('Processing %s %s files...' % (polarity, split))

    texts = []
    pattern = os.path.join(tgz_dir, split, polarity, '*')
    filenames = glob(pattern)
    if verbose == 2:
        for f in tqdm(filenames, leave=False):
            text = open(f).read()
            texts.append(text)
    else:
        for f in filenames:
            text = open(f).read()
            texts.append(text)

    dirname = os.path.dirname(processed_dir)
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)

    basename = '%s_%s_count.txt' % (split, polarity)
    out = os.path.join(processed_dir, basename)
    with open(out, 'wb') as out:
        text = str(len(texts)).encode('utf-8')
        out.write(text)

    basename = '%s_%s.txt' % (split, polarity)
    out = os.path.join(processed_dir, basename)
    with open(out, 'wb') as out:
        for text in texts:
            line = (text + '\n').encode('utf-8')
            out.write(line)


def _process(tgz_dir, processed_dir, verbose):
    for split in ['train', 'test']:
        for polarity in ['pos', 'neg']:
            _process_split_polarity(
                tgz_dir, processed_dir, split, polarity, verbose)
            _process_split_polarity(
                tgz_dir, processed_dir, split, polarity, verbose)


def _load_split_polarity(processed_dir, split, polarity, verbose):
    basename = '%s_%s_count.txt' % (split, polarity)
    filename = os.path.join(processed_dir, basename)
    count = int(open(filename).read())

    basename = '%s_%s.txt' % (split, polarity)
    filename = os.path.join(processed_dir, basename)
    texts = []
    if verbose == 2:
        for line in tqdm(open(filename), total=count, leave=False):
            texts.append(line.strip())
    else:
        for line in open(filename):
            texts.append(line.strip())

    return texts


def _load_split(processed_dir, split, verbose):
    pos = _load_split_polarity(processed_dir, split, 'pos', verbose)
    neg = _load_split_polarity(processed_dir, split, 'neg', verbose)
    data = _combine(pos, neg)
    texts, labels = list(zip(*data))
    return texts, np.array(labels, dtype='int64')


def _load(processed_dir, verbose):
    train = _load_split(processed_dir, 'train', verbose)
    val = _load_split(processed_dir, 'test', verbose)
    return train, val


def load_imdb(verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    processed_dir = os.path.join(dataset_dir, _PROCESSED_SUBDIR)
    if not os.path.exists(processed_dir):
        tgz_dir = os.path.join(dataset_dir, _TGZ_SUBDIR)
        if not os.path.exists(tgz_dir):
            tgz_basename = os.path.basename(_URL)
            local = os.path.join(dataset_dir, tgz_basename)
            download(_URL, local, verbose)
            _extract(dataset_dir, tgz_basename, verbose)
        _process(tgz_dir, processed_dir, verbose)
    return _load(processed_dir, verbose)
