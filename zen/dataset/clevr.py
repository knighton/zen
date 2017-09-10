import io
import json
import numpy as np
import os
from PIL import Image
import re
from time import time
from tqdm import tqdm
from zipfile import ZipFile

from ..model.data.dataset import Dataset
from ..model.data.training_data import TrainingData
from .util import download, get_dataset_dir


_DATASET_NAME = 'clevr'
_MAIN_URL = 'https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip'
_MAIN_PROCESSED_SUBDIR = 'main_processed'
_MAIN_SPLITS = ['train', 'val', 'test']
_COGENT_URL = 'https://s3-us-west-1.amazonaws.com/clevr/CLEVR_CoGenT_v1.0.zip'
_COGENT_PROCESSED_SUBDIR = 'cogent_processed'
_COGENT_ALL_SPLITS = ['trainA', 'valA', 'valB', 'testA', 'testB']
_COGENT_SAME_SPLITS = ['trainA', 'valA', 'testA']
_COGENT_DIFF_SPLITS = ['trainA', 'valB', 'testB']
_IMAGE_RE = re.compile('.+/images/.+.png')
_QUESTIONS_RE = re.compile('.+/questions/.+.json')


_SIZES = {
    (60, 40),
    (120, 80),
    (240, 160),
}


def _normalize_size(x):
    if x is None:
        x = 60, 40
    assert x in _SIZES, \
        'Size %s is not valid.  Possible values: %s.' % (size, _SIZES)
    return x


def _extract_image(zip_file, path):
    data = zip_file.open(path).read()
    image = Image.open(io.BytesIO(data))
    image = image.convert('RGB')
    assert image.size == (480, 320)
    _240 = image.resize((240, 160))
    _120 = image.resize((120, 80))
    _60 = image.resize((60, 40))
    return np.array(_240).tobytes(), np.array(_120).tobytes(), \
           np.array(_60).tobytes()


def _extract_images(zip_file, zip_paths, split, processed_dir, verbose):
    paths = list(filter(_IMAGE_RE.match, zip_paths))
    paths = list(filter(lambda s: split in s, paths))
    paths.sort()
    out_240 = os.path.join(processed_dir, 'images_240x160_%s.bin' % split)
    out_240 = open(out_240, 'wb')
    out_120 = os.path.join(processed_dir, 'images_120x80_%s.bin' % split)
    out_120 = open(out_120, 'wb')
    out_60 = os.path.join(processed_dir, 'images_60x40_%s.bin' % split)
    out_60 = open(out_60, 'wb')
    gen = paths
    if verbose == 2:
        gen = tqdm(gen, leave=False)
    for path in gen:
        _240, _120, _60 = _extract_image(zip_file, path)
        out_240.write(_240)
        out_120.write(_120)
        out_60.write(_60)


def _extract_questions(zip_file, zip_paths, split, processed_dir, verbose):
    paths = list(filter(_QUESTIONS_RE.match, zip_paths))
    paths = list(filter(lambda s: split in s, paths))
    assert len(paths) == 1
    path, = paths
    text = zip_file.open(path).read().decode('utf-8')
    x = json.loads(text)
    out = os.path.join(processed_dir, 'questions_%s.txt' % split)
    out = open(out, 'wb')
    gen = x['questions']
    if verbose == 2:
        gen = tqdm(gen, leave=False)
    for d in gen:
        image = d['image_index']
        question = d['question']
        answer = d.get('answer')
        d = {
            'image': image,
            'question': question,
            'answer': answer,
        }
        line = json.dumps(d, sort_keys=True) + '\n'
        out.write(line.encode('utf-8'))


def _extract_main_split(zip_file, zip_paths, split, processed_dir, verbose):
    if verbose:
        print('Extracting %s split...' % split)
        t0 = time()
    _extract_images(zip_file, zip_paths, split, processed_dir, verbose)
    _extract_questions(zip_file, zip_paths, split, processed_dir, verbose)
    if verbose:
        t = time() - t0
        print('...took %.3f sec.' % t)


def _extract(zip_filename, processed_dir, splits, verbose):
    os.mkdir(processed_dir)
    if verbose:
        print('Loading archive at %s...' % zip_filename)
        t0 = time()
    zip_file = ZipFile(zip_filename)
    zip_paths = zip_file.namelist()
    if verbose:
        t = time() - t0
        print('...took %.3f sec.' % t)
    for split in splits:
        _extract_main_split(zip_file, zip_paths, split, processed_dir, verbose)


class RamClevrDataset(Dataset):
    def __init__(self, images, image_ids, questions, answers):
        self.images = images
        self.image_ids = image_ids
        self.questions = questions
        self.answers = answers

    def get_num_samples(self):
        return len(self.image_ids)

    def get_sample(self, index):
        image_id = self.image_ids[index]
        image = self.images[image_id]
        question = self.questions[index]
        answer = self.answers[index]
        return (image, question), answer


def _load_split(processed_dir, split, size, storage, verbose):
    assert storage == 'ram'  # XXX
    assert len(size) == 2
    filename = os.path.join(
        processed_dir, 'images_%dx%d_%s.bin' % (size[0], size[1], split))
    images = np.load(filename).astype('float32')
    filename = os.path.join(processed_dir, 'questions_%s.txt' % split)
    answers = []
    image_ids = []
    questions = []
    for line in open(filename):
        x = json.loads(line)
        answer = x['answer']
        answers.append(answer)
        image_id = x['image']
        image_ids.append(image_id)
        question = x['question']
        questions.append(question)
    return RamClevrDataset(images, image_ids, questions, answers)


def _load(processed_dir, splits, size, storage, verbose):
    assert len(splits) == 3
    train = _load_split(processed_dir, splits[0], size, storage, verbose)
    val = _load_split(processed_dir, splits[1], size, storage, verbose)
    return TrainingData(train, val)


def _load_clevr_cogent_process(verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    processed_dir = os.path.join(dataset_dir, _COGENT_PROCESSED_SUBDIR)
    if not os.path.exists(processed_dir):
        zip_filename = os.path.join(dataset_dir, os.path.basename(_COGENT_URL))
        if not os.path.exists(zip_filename):
            download(_COGENT_URL, zip_filename, verbose)
        _extract(zip_filename, processed_dir, _COGENT_ALL_SPLITS, verbose)
    return processed_dir


def load_clevr_main(verbose=2, storage='ram', size=None):
    assert storage in {'ram', 'disk'}
    size = _normalize_size(size)
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    processed_dir = os.path.join(dataset_dir, _MAIN_PROCESSED_SUBDIR)
    if not os.path.exists(processed_dir):
        zip_filename = os.path.join(dataset_dir, os.path.basename(_MAIN_URL))
        if not os.path.exists(zip_filename):
            download(_MAIN_URL, zip_filename, verbose)
        _extract(zip_filename, processed_dir, _MAIN_SPLITS, verbose)
    return _load(processed_dir, _MAIN_SPLITS, size, storage, verbose)


def load_clevr_cogent_same(verbose=2, storage='ram', size=None):
    assert storage in {'ram', 'disk'}
    size = _normalize_size(size)
    processed_dir = _load_clevr_cogent_process(verbose)
    return _load(processed_dir, _COGENT_SAME_SPLITS, size, storage, verbose)


def load_clevr_cogent_diff(verbose=2, storage='ram', size=None):
    assert storage in {'ram', 'disk'}
    size = _normalize_size(size)
    processed_dir = _load_clevr_cogent_process(verbose)
    return _load(processed_dir, _COGENT_DIFF_SPLITS, size, storage, verbose)
