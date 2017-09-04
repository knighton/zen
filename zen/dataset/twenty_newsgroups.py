from glob import glob
import json
import numpy as np
import os
from random import shuffle
import tarfile
from tqdm import tqdm

from .util import download, get_dataset_dir


_DATASET_NAME = 'twenty_newsgroups'
_URL = 'http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz'
_PROCESSED_SUBDIR = 'processed'


def _process(tgz_filename, processed_dir, verbose=2):
    if os.path.exists(processed_dir):
        return
    os.mkdir(processed_dir)
    tar = tarfile.open(tgz_filename, 'r:gz')
    infos = tar.getmembers()
    filename2out = {}
    if verbose == 2:
        infos = tqdm(infos, leave=False)
    for info in infos:
        pieces = info.name.split(os.path.sep)
        if len(pieces) != 3:
            continue
        split = pieces[0]
        split = split[split.rindex('-') + 1:]
        newsgroup = pieces[1]
        filename = os.path.join(
            processed_dir, '%s_%s.txt' % (newsgroup, split))
        data = tar.extractfile(info).read()
        text = data.decode('latin-1')
        line = json.dumps(text) + '\n'
        out = filename2out.get(filename)
        if out is None:
            out = open(filename, 'wb')
            filename2out[filename] = out
        out.write(line.encode('utf-8'))


def _get_newsgroup_id(newsgroup2id, newsgroup):
    id_ = newsgroup2id.get(newsgroup)
    if id_ is None:
        id_ = len(newsgroup2id)
        newsgroup2id[newsgroup] = id_
    return id_


class TwentyNewsgroups(object):
    def __init__(self, train, test, newsgroup2id):
        self.train = train
        self.test = test
        self.newsroup2id = newsgroup2id
        self.newsgroups = [None] * len(newsgroup2id)
        for newsgroup, id_ in newsgroup2id.items():
            self.newsgroups[id_] = newsgroup


def _load(processed_dir):
    pattern = os.path.join(processed_dir, '*')
    filenames = glob(pattern)
    filenames.sort()
    newsgroup2id = {}
    train_samples = []
    test_samples = []
    for filename in filenames:
        basename = os.path.basename(filename)
        assert basename.endswith('.txt')
        text = basename[:-4]
        newsgroup, split = text.split('_')
        newsgroup_id = _get_newsgroup_id(newsgroup2id, newsgroup)
        assert split in {'train', 'test'}
        samples = train_samples if split == 'train' else test_samples
        for line in open(filename):
            text = json.loads(line)
            sample = text, newsgroup_id
            samples.append(sample)
    shuffle(train_samples)
    train_texts, train_classes = zip(*train_samples)
    train_classes = np.array(train_classes)
    shuffle(test_samples)
    test_texts, test_classes = zip(*test_samples)
    test_classes = np.array(test_classes)
    train = train_texts, train_classes
    test = test_texts, test_classes
    return TwentyNewsgroups(train, test, newsgroup2id)


def load_twenty_newsgroups(verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    processed_dir = os.path.join(dataset_dir, _PROCESSED_SUBDIR)
    if not os.path.exists(processed_dir):
        tgz_filename = os.path.join(dataset_dir, os.path.basename(_URL))
        download(_URL, tgz_filename, verbose)
        _process(tgz_filename, processed_dir, verbose)
    return _load(processed_dir)
