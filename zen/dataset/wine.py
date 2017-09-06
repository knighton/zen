from collections import Counter
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


def _observe(rows):
    means = []
    stds = []
    for i in range(rows.shape[1]):
        means.append(rows[:, i].mean())
        stds.append(rows[:, i].std())
    return means, stds


def _do_scale(rows, means, stds):
    for i in range(rows.shape[1]):
        rows[:, i] -= means[i]
        rows[:, i] /= stds[i]


def _blend(red_samples, white_samples, val_frac, scale):
    samples = red_samples + white_samples
    shuffle(samples)
    x, y = zip(*samples)
    x = np.array(x, dtype='float32')
    y = np.array(y, dtype='float32')
    split = int(len(x) * val_frac)
    x_train = x[split:]
    x_val = x[:split]
    y_train = y[split:]
    y_val = y[:split]
    if scale:
        means, stds = _observe(x_train)
        _do_scale(x_train, means, stds)
        _do_scale(x_val, means, stds)
    return (x_train, y_train), (x_val, y_val)


def _load_color(remote, y, verbose):
    x = []
    for line in _get(remote, verbose):
        values = list(map(float, line.split(';')))
        x.append(values)
    return list(zip(x, [y] * len(x)))


def _load_quality(remote, verbose):
    x = []
    y = []
    for line in _get(remote, verbose):
        values = list(map(float, line.split(';')))
        x.append(values[:-1])
        y.append(values[-1] / 10.)
    return list(zip(x, y))


def load_wine_color(val_frac=0.2, scale=True, verbose=2):
    red = _load_color(_RED, 1, verbose)
    white = _load_color(_WHITE, 0, verbose)
    data = _blend(red, white, val_frac, scale)
    if verbose:
        print('Train: %d samples.' % len(data[0][0]))
        d = Counter(data[0][1])
        print('       %d red, %d white.' % (d[1.], d[0.]))
        print('Val:   %d samples.' % len(data[1][0]))
        d = dict(Counter(data[1][1]))
        print('       %d red, %d white.' % (d[1.], d[0.]))
    return data


def load_wine_quality(val_frac=0.2, scale=True, verbose=2):
    red = _load_quality(_RED, verbose)
    white = _load_quality(_WHITE, verbose)
    return _blend(red, white, val_frac, scale)
