import numpy as np
import os
from random import shuffle
from time import time
from tqdm import tqdm
from zipfile import ZipFile

from .util import download, get_dataset_dir


_DATASET_NAME = 'sentiment140'
_URL = 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
_ZIP_PATH = 'training.1600000.processed.noemoticon.csv'
_ENCODINGS = ['utf-8', 'latin-1']
_PROCESSED_SUBDIR = 'processed'
_POS_BASENAME = 'pos.txt'
_NEG_BASENAME = 'neg.txt'


def _write_tweets(tweets, filename, verbose):
    with open(filename, 'wb') as out:
        if verbose == 2:
            tweets = tqdm(tweets, leave=False)
        for tweet in tweets:
            assert '\n' not in tweet
            line = (tweet + '\n').encode('utf-8')
            out.write(line)


def _process(local, processed_dir, verbose):
    os.mkdir(processed_dir)
    zip_file = ZipFile(local)
    lines = zip_file.open(_ZIP_PATH).readlines()
    if verbose == 2:
        lines = tqdm(lines, leave=False)
    pos = []
    neg = []
    for line in lines:
        for encoding in _ENCODINGS:
            try:
                line = line.decode(encoding)
            except:
                pass
        ss = line[1:-1].split('","')
        assert len(ss) == 6
        sent = int(ss[0])
        if sent == 4:
            tweets = pos
        elif sent == 0:
            tweets = neg
        else:
            assert False
        tweet = ss[5]
        tweets.append(tweet)
    filename = os.path.join(processed_dir, _POS_BASENAME)
    _write_tweets(pos, filename, verbose)
    filename = os.path.join(processed_dir, _NEG_BASENAME)
    _write_tweets(neg, filename, verbose)


def _read_tweets(filename, verbose):
    tweets = []
    lines = open(filename)
    if verbose == 2:
        lines = tqdm(lines, leave=False)
    tweets = []
    for line in lines:
        tweet = line[:-1]
        tweets.append(tweet)
    return tweets


def _load(processed_dir, val_frac, verbose):
    filename = os.path.join(processed_dir, _POS_BASENAME)
    pos = _read_tweets(filename, verbose)
    filename = os.path.join(processed_dir, _NEG_BASENAME)
    neg = _read_tweets(filename, verbose)
    samples = list(map(lambda tweet: (tweet, 1), pos)) + \
              list(map(lambda tweet: (tweet, 0), neg))
    if verbose:
        t0 = time()
    shuffle(samples)
    if verbose:
        t = time() - t0
        print('Shuffling %d items took %.3f sec.' % (len(samples), t))
    x, y = list(zip(*samples))
    split = int(len(x) * val_frac)
    x_train = x[split:]
    x_val = x[:split]
    y_train = y[split:]
    y_val = y[:split]
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    return (x_train, y_train), (x_val, y_val)


def load_sentiment140(val_frac=0.2, verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    processed_dir = os.path.join(dataset_dir, _PROCESSED_SUBDIR)
    if not os.path.exists(processed_dir):
        zip_basename = os.path.basename(_URL)
        local = os.path.join(dataset_dir, zip_basename)
        download(_URL, local, verbose)
        _process(local, processed_dir, verbose)
    return _load(processed_dir, val_frac, verbose)
