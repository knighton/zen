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
    _240 = np.array(_240)
    _240 = np.rollaxis(_240, 2)
    _120 = image.resize((120, 80))
    _120 = np.array(_120)
    _120 = np.rollaxis(_120, 2)
    _60 = image.resize((60, 40))
    _60 = np.array(_60)
    _60 = np.rollaxis(_60, 2)
    return _240.tobytes(), _120.tobytes(), _60.tobytes()


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
        print('Loading archive at %s' % zip_filename)
        t0 = time()
    zip_file = ZipFile(zip_filename)
    zip_paths = zip_file.namelist()
    if verbose:
        t = time() - t0
        print('...took %.3f sec.' % t)
    for split in splits:
        _extract_main_split(zip_file, zip_paths, split, processed_dir, verbose)


def _process(dataset_name, processed_subdir, url, all_splits, verbose):
    dataset_dir = get_dataset_dir(dataset_name)
    processed_dir = os.path.join(dataset_dir, processed_subdir)
    if not os.path.exists(processed_dir):
        filename = os.path.join(dataset_dir, os.path.basename(url))
        if not os.path.exists(filename):
            download(url, filename, verbose)
        _extract(filename, processed_dir, all_splits, verbose)
    return processed_dir


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


def _load_split(question_pipe, answer_pipe, processed_dir, split,
                is_train_split, size, storage, verbose):
    assert storage == 'ram'  # XXX
    assert len(size) == 2
    filename = os.path.join(
        processed_dir, 'images_%dx%d_%s.bin' % (size[0], size[1], split))
    print('Reading %s...' % filename)
    data = open(filename, 'rb').read()
    images = np.fromstring(data, dtype='uint8')
    images = images.astype('float32')
    images = images.reshape(-1, 3, size[0], size[1])
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
    image_ids = np.array(image_ids, dtype='int32')
    if is_train_split:
        questions = question_pipe.fit_transform(questions)
        answers = answer_pipe.fit_transform(answers)
    else:
        questions = question_pipe.transform(questions)
        answers = answer_pipe.transform(answers)
    return RamClevrDataset(images, image_ids, questions, answers)


def _load(question_pipe, answer_pipe, processed_dir, splits, size, storage,
          verbose):
    assert len(splits) == 3
    train = _load_split(question_pipe, answer_pipe, processed_dir, splits[0],
                        True, size, storage, verbose)
    val = _load_split(question_pipe, answer_pipe, processed_dir, splits[1],
                      False, size, storage, verbose)
    return TrainingData(train, val)


def load_clevr_main(question_pipe, answer_pipe, storage='ram', size=None,
                    verbose=2):
    assert storage in {'ram', 'disk'}
    size = _normalize_size(size)
    processed_dir = _process(_DATASET_NAME, _MAIN_PROCESSED_SUBDIR, _MAIN_URL,
                             _MAIN_SPLITS, verbose)
    return _load(question_pipe, answer_pipe, processed_dir, _MAIN_SPLITS, size,
                 storage, verbose)


def load_clevr_cogent_same(question_pipe, answer_pipe, storage='ram', size=None,
                           verbose=2):
    assert storage in {'ram', 'disk'}
    size = _normalize_size(size)
    processed_dir = _process(_DATASET_NAME, _COGENT_PROCESSED_SUBDIR,
                             _COGENT_URL, _COGENT_ALL_SPLITS, verbose)
    return _load(question_pipe, answer_pipe, processed_dir, _COGENT_SAME_SPLITS,
                 size, storage, verbose)


def load_clevr_cogent_diff(question_pipe, answer_pipe, storage='ram', size=None,
                           verbose=2):
    assert storage in {'ram', 'disk'}
    size = _normalize_size(size)
    processed_dir = _process(_DATASET_NAME, _COGENT_PROCESSED_SUBDIR,
                             _COGENT_URL, _COGENT_ALL_SPLITS, verbose)
    return _load(question_pipe, answer_pipe, processed_dir, _COGENT_DIFF_SPLITS,
                 size, storage, verbose)
