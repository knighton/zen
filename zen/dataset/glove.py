import numpy as np
import os
from time import time
from tqdm import tqdm

from .util import download, get_dataset_dir


_NAME2URL = {}


def _add(url):
    name = url[url.rindex('/') + 1:url.rindex('.')]
    _NAME2URL[name] = url


# Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB
# download): glove.42B.300d.zip
_add('http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip')

# Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download):
# glove.840B.300d.zip
_add('http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip')

# Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 300d vectors,
# 822 MB download): glove.6B.zip
_add('http://nlp.stanford.edu/data/wordvecs/glove.6B.zip')

# Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 200d vectors, 1.42 GB
# download): glove.twitter.27B.zip
_add('http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip')


def _unzip(dataset_dir, local_zip, verbose):
    local_txt = local_zip[:-3] + 'txt'
    if os.path.exists(local_txt):
        return local_txt
    cmd = 'cd %s; unzip %s' % (dataset_dir, local_zip)
    if verbose:
        t0 = time()
    assert not os.system(cmd)
    if verbose:
        t = time() - t0
        print('Extracting archive took %.3f sec.' % t)
    return local_txt


def _wc(dataset_dir, local_txt, verbose):
    count_txt = local_txt + '.count.txt'
    if os.path.exists(count_txt):
        return
    if verbose:
        t0 = time()
    gen = open(local_txt)
    if verbose == 2:
        gen = tqdm(gen, leave=False)
    count = 0
    for line in gen:
        count += 1
    with open(count_txt, 'wb') as out:
        out.write(str(count).encode('utf-8'))
    if verbose:
        t = time() - t0
        print('Vector count took %.3f sec.' % t)


def _fetch(which, verbose):
    dataset_dir = get_dataset_dir('glove')
    remote = _NAME2URL[which]
    basename = os.path.basename(remote)
    local_zip = os.path.join(dataset_dir, basename)
    download(remote, local_zip, verbose)
    local_txt = _unzip(dataset_dir, local_zip, verbose)
    _wc(dataset_dir, local_txt, verbose)
    return local_txt


def _read_text(txt_filename, max_vocab_size, verbose):
    count = int(open(txt_filename + '.count.txt').read())
    word2vector = {}
    gen = open(txt_filename)
    if max_vocab_size is not None:
        count = min(count, max_vocab_size)
    if verbose == 2:
        gen = tqdm(gen, total=count, leave=False)
    vector_len = None
    for i, line in enumerate(gen):
        if i == count:
            break
        tokens = line.split()
        if vector_len is None:
            vector_len = len(tokens) - 1
        word = tokens[0]
        floats = list(map(float, tokens[-vector_len:]))
        vector = np.array(floats, dtype='float32')
        word2vector[word] = vector
    return word2vector


def load_glove(which='glove.840B.300d', max_vocab_size=None, verbose=2):
    txt_filename = _fetch(which, verbose)
    return _read_text(txt_filename, max_vocab_size, verbose)
