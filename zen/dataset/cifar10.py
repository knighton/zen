import numpy as np
import os
import pickle
from random import shuffle
import tarfile
from tqdm import tqdm

from .util import download, get_dataset_dir


_DATASET_NAME = 'cifar'
_CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


class CIFAR10(object):
    def __init__(self, x, y, classes):
        self.x = x
        self.y = y
        self.classes = classes


def _load_x_y(tar, verbose):
    if verbose == 2:
        bar = tqdm(total=5, leave=False)
    xx = []
    yy = []
    for info in tar.getmembers():
        if not info.isreg():
            continue
        if not info.path.startswith('cifar-10-batches-py/data_batch_'):
            continue
        data = tar.extractfile(info).read()
        obj = pickle.loads(data, encoding='bytes')
        x = obj[b'data']
        x = x.reshape(10000, 3, 32, 32)
        x = x.astype('float32') / 255.
        y = np.array(obj[b'labels'], dtype='int64')
        xx.append(x)
        yy.append(y)
        if verbose == 2:
            bar.update(1)
    if verbose == 2:
        bar.close()
    return np.vstack(xx), np.hstack(yy)


def _load_classes(tar):
    path = 'cifar-10-batches-py/batches.meta'
    data = tar.extractfile(path).read()
    obj = pickle.loads(data, encoding='bytes')
    classes = obj[b'label_names']
    return list(map(lambda s: s.decode('utf-8'), classes))


def load_cifar10(verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    filename = os.path.join(dataset_dir, os.path.basename(_CIFAR10_URL))
    download(_CIFAR10_URL, filename, verbose)
    tar = tarfile.open(filename, 'r:gz')
    x, y = _load_x_y(tar, verbose)
    classes = _load_classes(tar)
    tar.close()
    return CIFAR10(x, y, classes)
