import gzip
import os
import pickle

from .util import download, get_dataset_dir


_DATASET_NAME = 'mnist'
_URL = 'https://s3.amazonaws.com/img-datasets/mnist.pkl.gz'


def load_mnist(verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    filename = os.path.join(dataset_dir, os.path.basename(_URL))
    download(_URL, filename, verbose)
    (x_train, y_train), (x_test, y_test) = \
        pickle.load(gzip.open(filename), encoding='latin1')
    x_train = x_train.astype('float32') / 255.
    y_train = y_train.astype('int64')
    x_test = x_test.astype('float32') / 255.
    y_test = y_test.astype('int64')
    return (x_train, y_train), (x_test, y_test)
