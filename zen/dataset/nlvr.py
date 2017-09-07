from collections import defaultdict
from glob import glob
import json
import numpy as np
import os
from PIL import Image
from time import time
from tqdm import tqdm

from .util import get_dataset_dir


def _clone_dataset_repo(dataset_dir):
    if not os.path.exists(dataset_dir):
        repo = 'https://github.com/clic-lab/nlvr'
        cmd = 'mkdir -p %s; cd %s; git clone %s' % \
            (dataset_dir, dataset_dir, repo)
        assert not os.system(cmd)


def load_image(f):
    im = Image.open(f)
    im = im.convert('RGB')
    im = im.resize((200, 50))
    im = np.array(im, dtype=np.float32)
    im /= 255
    im = np.rollaxis(im, 2)
    return im


def _load_split(dataset_dir, split, verbose=2):
    assert split in {'train', 'dev', 'test'}
    assert verbose in {0, 1, 2}

    repo_dir = os.path.join(dataset_dir, 'nlvr')

    samples = []
    f = os.path.join(repo_dir, split, '%s.json' % split)
    for line in open(f):
        j = json.loads(line)
        sentence = j['sentence']
        label = j['label']
        name = j['identifier']
        samples.append((name, sentence, label))
    if 1 <= verbose:
        print('%d %s sentence/label pairs.' % (len(samples), split))

    pattern = os.path.join(repo_dir, split, 'images', '*', '*')
    ff = glob(pattern)
    if 1 <= verbose:
        print('%d %s images.' % (len(ff), split))

    if 1 <= verbose:
        print('Loading images into memory...')
    filename2image = {}
    if verbose == 2:
        t0 = time()
        for f in tqdm(ff):
            image = load_image(f)
            filename2image[f] = image
        t = time() - t0
    else:
        t0 = time()
        for f in ff:
            image = load_image(f)
            filename2image[f] = image
        t = time() - t0
    print('...took %.3f sec.' % t)

    name2filenames = defaultdict(list)
    for f in ff:
        a = f.rfind(split) + len(split) + 1
        z = f.rfind('-')
        name = f[a:z]
        name2filenames[name].append(f)

    rr = []
    for name, sentence, label in samples:
        for filename in name2filenames[name]:
            rr.append((filename, sentence, label))

    np.random.shuffle(rr)

    filenames, sentences, labels = zip(*rr)
    images = np.stack(map(lambda f: filename2image[f], filenames))

    if verbose:
        print('Images (%s): %s.' % (split, images.shape,))

    return (images, sentences), labels


def load_nlvr(verbose=2):
    dataset_dir = get_dataset_dir('nlvr')
    _clone_dataset_repo(dataset_dir)
    train = _load_split(dataset_dir, 'train', verbose)
    val = _load_split(dataset_dir, 'dev', verbose)
    return train, val
