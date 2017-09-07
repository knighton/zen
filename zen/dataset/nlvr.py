from collections import defaultdict
from glob import glob
import json
import numpy as np
import os
from PIL import Image
from time import time
from tqdm import tqdm

from .util import get_dataset_dir


_DATASET_NAME = 'nlvr'
_PROCESSED_SUBDIR = 'processed'
_REPO = 'https://github.com/clic-lab/nlvr'


def _clone_dataset_repo(dataset_dir):
    os.mkdir(dataset_dir)
    cmd = 'mkdir -p %s; cd %s; git clone %s' % \
        (dataset_dir, dataset_dir, _REPO)
    assert not os.system(cmd)


def load_image(f):
    im = Image.open(f)
    im = im.convert('RGB')
    assert im.size == (400, 100)
    im = im.resize((200, 50))
    im = np.array(im, dtype=np.uint8)
    im = np.rollaxis(im, 2)
    return im


def _process_split(dataset_dir, split, verbose=2):
    assert split in {'train', 'dev', 'test'}
    assert verbose in {0, 1, 2}

    repo_dir = os.path.join(dataset_dir, os.path.basename(_REPO))

    samples = []
    filename = os.path.join(repo_dir, split, '%s.json' % split)
    for line in open(filename):
        j = json.loads(line)
        sentence = j['sentence']
        label = j['label']
        name = j['identifier']
        samples.append((name, sentence, label))
    if 1 <= verbose:
        print('%d %s sentence/label pairs.' % (len(samples), split))

    pattern = os.path.join(repo_dir, split, 'images', '*', '*')
    filenames = glob(pattern)
    if 1 <= verbose:
        print('%d %s images.' % (len(filenames), split))

    filename2image = {}
    if verbose == 2:
        filenames = tqdm(filenames, leave=False)
    if verbose:
        t0 = time()
    for filename in filenames:
        image = load_image(filename)
        filename2image[filename] = image
    if verbose:
        t = time() - t0
        print('Loading %s images into memory took %.3f sec.' % (split, t))

    name2filenames = defaultdict(list)
    for filename in filenames:
        a = filename.rfind(split) + len(split) + 1
        z = filename.rfind('-')
        name = filename[a:z]
        name2filenames[name].append(filename)

    data = []
    for name, sentence, label in samples:
        for filename in name2filenames[name]:
            data.append((filename, sentence, label))

    np.random.shuffle(data)

    processed_dir = os.path.join(dataset_dir, _PROCESSED_SUBDIR)

    if verbose:
        t0 = time()
    out = os.path.join(processed_dir, '%s_images.npy' % split)
    filenames, sentences, labels = zip(*data)
    images = np.stack(map(lambda f: filename2image[f], filenames))
    np.save(out, images)
    if verbose:
        t = time() - t0
        print('Writing %s images %s took %.3f sec.' % (split, images.shape, t))

    out = os.path.join(processed_dir, '%s_text.txt' % split)
    with open(out, 'wb') as out:
        if verbose == 2:
            data = tqdm(data, leave=False)
        for filename, sentence, label in data:
            x = {
                'filename': filename,
                'sentence': sentence,
                'label': label,
            }
            line = json.dumps(x) + '\n'
            out.write(line.encode('utf-8'))

    out = os.path.join(processed_dir, '%s_count.txt' % split)
    with open(out, 'wb') as out:
        text = str(len(data))
        out.write(text.encode('utf-8'))


def _load_split(processed_dir, split, verbose):
    in_ = os.path.join(processed_dir, '%s_images.npy' % split)
    if verbose:
        t0 = time()
    images = np.load(in_)
    if verbose:
        t = time() - t0
        print('Reading %s images %s took %.3f sec.' % (split, images.shape, t))

    in_ = os.path.join(processed_dir, '%s_count.txt' % split)
    count = int(open(in_).read())

    in_ = os.path.join(processed_dir, '%s_text.txt' % split)
    in_ = open(in_)
    if verbose == 2:
        in_ = tqdm(in_, total=count, leave=False)
    sentences = []
    labels = []
    for line in in_:
        x = json.loads(line)
        sentence = x['sentence']
        sentences.append(sentence)
        label = x['label']
        labels.append(label)
    return images, sentences, labels


def load_nlvr(verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    processed_dir = os.path.join(dataset_dir, _PROCESSED_SUBDIR)
    if not os.path.exists(processed_dir):
        repo_dir = os.path.join(dataset_dir, os.path.basename(_REPO))
        if not os.path.exists(repo_dir):
            _clone_dataset_repo(dataset_dir)
        os.mkdir(processed_dir)
        _process_split(dataset_dir, 'train', verbose)
        _process_split(dataset_dir, 'dev', verbose)
    train = _load_split(processed_dir, 'train', verbose)
    val = _load_split(processed_dir, 'dev', verbose)
    return train, val
