import numpy as np
import os
import pickle
from random import shuffle
import tarfile
from tqdm import tqdm

from .util import download, get_dataset_dir


_DATASET_NAME = 'cifar'
_CIFAR100_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'


def _load_split(tar, num_classes, split):
    path = 'cifar-100-python/%s' % split
    data = tar.extractfile(path).read()
    obj = pickle.loads(data, encoding='bytes')
    x = obj[b'data']
    x = x.reshape(x.shape[0], 3, 32, 32)
    x = x.astype('float32') / 255.
    key = {
        20: b'coarse_labels',
        100: b'fine_labels',
    }[num_classes]
    y = np.array(obj[key], dtype='int64')
    return x, y


def _load_classes(tar, num_classes):
    info = tar.getmember('cifar-100-python/meta')
    data = tar.extractfile(info).read()
    obj = pickle.loads(data, encoding='bytes')
    if num_classes == 20:
        key = b'coarse_label_names'
    elif num_classes == 100:
        key = b'fine_label_names'
    else:
        assert False
    classes = obj[key]
    return list(map(lambda s: s.decode('utf-8'), classes))


class CIFAR100(object):
    def __init__(self, train, val, classes):
        self.train = train
        self.val = val
        self.classes = classes


def load_cifar100(num_classes=100, verbose=2):
    assert num_classes in {20, 100}
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    filename = os.path.join(dataset_dir, os.path.basename(_CIFAR100_URL))
    download(_CIFAR100_URL, filename, verbose)
    tar = tarfile.open(filename, 'r:gz')
    train = _load_split(tar, num_classes, 'train')
    val = _load_split(tar, num_classes, 'test')
    classes = _load_classes(tar, num_classes)
    tar.close()
    return CIFAR100(train, val, classes)
