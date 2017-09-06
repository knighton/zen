import numpy as np
import os
from random import shuffle

from .util import download, get_dataset_dir


_RED = 'http://archive.ics.uci.edu/ml/machine-learning-databases/' + \
       'wine-quality/winequality-red.csv'
_WHITE = 'http://archive.ics.uci.edu/ml/machine-learning-databases/' + \
         'wine-quality/winequality-white.csv'


def _get(remote, verbose):
    dataset_dir = get_dataset_dir('wine_quality')
    local = os.path.join(dataset_dir, os.path.basename(remote))
    download(remote, local, verbose)
    return open(local).readlines()[1:]


def _scale_samples(rows):
    columns = list(zip(*rows))
    for i, column in enumerate(columns):
        column = np.array(column)
        print('Column %d: mean %.3f std %.3f' %
              (i, column.mean(), column.std()))
        column -= column.mean()
        column /= column.std()
        columns[i] = column
    return list(zip(*columns))


def _blend(red_samples, white_samples):
    samples = red_samples + white_samples
    shuffle(samples)
    x, y = zip(*samples)
    return np.array(x, dtype='float32'), np.array(y, dtype='float32')


def _load_color(remote, y, scale, verbose):
    x = []
    for line in _get(remote, verbose):
        values = list(map(float, line.split(';')))
        x.append(values)
    if scale:
        x = _scale_samples(x)
    return list(zip(x, [y] * len(x)))


def _load_quality(remote, scale, verbose):
    x = []
    y = []
    for line in _get(remote, verbose):
        values = list(map(float, line.split(';')))
        x.append(values[:-1])
        y.append(values[-1])
    if scale:
        x = _scale_samples(x)
    return list(zip(x, y))


def load_wine_color(scale=True, verbose=2):
    red = _load_color(_RED, 1, scale, verbose)
    white = _load_color(_WHITE, 0, scale, verbose)
    return _blend(red, white)


def load_wine_quality(scale=True, verbose=2):
    red = _load_quality(_RED, scale, verbose)
    white = _load_quality(_WHITE, scale, verbose)
    return _blend(red, white)
