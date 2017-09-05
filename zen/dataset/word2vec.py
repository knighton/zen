import numpy as np
import os
from tqdm import tqdm

from .util import download, get_dataset_dir


_DATASET_NAME = 'word2vec'

_REMOTE = 'https://s3.amazonaws.com/dl4j-distribution/' + \
          'GoogleNews-vectors-negative300.bin.gz'


def _fetch(remote):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    basename_gz = os.path.basename(_REMOTE)
    local_gz = os.path.join(dataset_dir, basename_gz)
    basename_bin = basename_gz[:basename_gz.rindex('.')]
    local_bin = os.path.join(dataset_dir, basename_bin)
    if not os.path.exists(local_bin):
        if not os.path.exists(local_gz):
            download(_REMOTE, local_gz, verbose)
        cmd = 'cd %s; gunzip --keep %s' % (dataset_dir, basename_gz)
        assert not os.system(cmd)
    return local_bin


def _read_binary_word(file_):
    word = []
    while True:
        c = file_.read(1)
        if c == b' ':
            break
        word.append(c)
    return b''.join(word).decode('utf-8')


def _read_binary(bin_filename, max_vocab_size, verbose):
    file_ = open(bin_filename, 'rb')
    num_vectors, vector_len = map(int, file_.readline().split())
    if max_vocab_size is not None:
        num_vectors = min(num_vectors, max_vocab_size)
    gen = range(num_vectors)
    if verbose == 2:
        gen = tqdm(gen)
    word2vector = {}
    for i in gen:
        word = _read_binary_word(file_)
        vector = np.fromstring(file_.read(4 * vector_len))
        word2vector[word] = vector
    return word2vector


def load_word2vec(max_vocab_size=None, verbose=2):
    bin_filename = _fetch(_REMOTE)
    return _read_binary(bin_filename, max_vocab_size, verbose)
