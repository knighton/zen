import numpy as np
import os
import pickle
from random import shuffle
import tarfile
from tqdm import tqdm

from .cifar10 import load_cifar10
from .cifar100 import load_cifar100


def _split(x, y, val_frac):
    samples = list(zip(x, y))
    shuffle(samples)
    split = int(len(samples) * val_frac)
    x_train = x[split:]
    x_val = x[:split]
    y_train = y[split:]
    y_val = y[:split]
    return (x_train, y_train), (x_val, y_val)


class CIFAR(object):
    def __init__(self, train, val, labels):
        self.train = train
        self.val = val
        self.labels = labels


def load_cifar(num_classes=10, cifar10_val_frac=0.2, verbose=2):
    if num_classes == 10:
        assert 0. <= cifar10_val_frac <= 1.
        data = load_cifar10(verbose)
        train, val = _split(data.x, data.y, cifar10_val_frac)
        data = CIFAR(train, val, data.labels)
        data.val_frac = cifar10_val_frac
    elif num_classes in {20, 100}:
        data = load_cifar100(num_classes, verbose)
        data = CIFAR(data.train, data.val, data.labels)
    else:
        assert False
    return data
