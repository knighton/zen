from io import BytesIO
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
_IMAGE_SHAPES = [
    (60, 40),
    (120, 80),
    (240, 160),
]


def _normalize_image_shape(shape, possible_image_shapes):
    if shape is None:
        shape = 60, 40
    assert shape in possible_image_shapes, \
        'Image shape %s is not available.  Possible values: %s.' % \
            (shape, possible_image_shapes)
    return shape


def _get_processed_images_filename(processed_dir, split, shape):
    basename = 'images_%dx%d_%s.bin' % (shape[0], shape[1], split)
    return os.path.join(processed_dir, basename)


def _process_image(zip_file, path, to_shapes, files):
    data = zip_file.open(path).read()
    image = Image.open(BytesIO(data))
    image = image.convert('RGB')
    assert image.size == (480, 320)
    for to_shape, file_ in zip(to_shapes, files):
        x = image.resize(to_shape)
        x = np.array(x)
        x = np.rollaxis(x, 2)
        file_.write(x.tobytes())


def _process_images(zip_file, zip_paths, processed_dir, split, image_shapes,
                    verbose):
    paths = list(filter(_IMAGE_RE.match, zip_paths))
    paths = list(filter(lambda s: split in s, paths))
    paths.sort()
    files = []
    for shape in image_shapes:
        filename = _get_processed_images_filename(processed_dir, split, shape)
        files.append(open(filename, 'wb'))
    if verbose == 2:
        paths = tqdm(paths, leave=False)
    for path in paths:
        _process_image(zip_file, path, image_shapes, files)
    for file_ in files:
        file_.close()


def _process_questions(zip_file, zip_paths, processed_dir, split, verbose):
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


def _process_split(zip_file, zip_paths, processed_dir, split, image_shapes,
                   verbose):
    if verbose:
        print('Extracting %s split...' % split)
        t0 = time()
    _process_images(zip_file, zip_paths, processed_dir, split, image_shapes,
                    verbose)
    _process_questions(zip_file, zip_paths, processed_dir, split, verbose)
    if verbose:
        t = time() - t0
        print('...took %.3f sec.' % t)


def _process(zip_filename, processed_dir, splits, image_shapes, verbose):
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
        _process_split(zip_file, zip_paths, processed_dir, split, image_shapes,
                       verbose)


def _ready(dataset_name, processed_subdir, url, all_splits, image_shapes,
           verbose):
    dataset_dir = get_dataset_dir(dataset_name)
    processed_dir = os.path.join(dataset_dir, processed_subdir)
    if not os.path.exists(processed_dir):
        filename = os.path.join(dataset_dir, os.path.basename(url))
        if not os.path.exists(filename):
            download(url, filename, verbose)
        _process(filename, processed_dir, all_splits, image_shapes, verbose)
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
                is_train_split, image_shape, storage, verbose):
    assert storage == 'ram'  # XXX
    assert len(image_shape) == 2
    filename = _get_processed_images_filename(processed_dir, split, image_shape)
    print('Reading %s...' % filename)
    data = open(filename, 'rb').read()
    images = np.fromstring(data, dtype='uint8')
    images = images.astype('float32')
    images = images.reshape(-1, 3, image_shape[0], image_shape[1])
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


def _load(question_pipe, answer_pipe, processed_dir, splits, image_shape,
          storage, verbose):
    assert len(splits) == 3
    train = _load_split(question_pipe, answer_pipe, processed_dir, splits[0],
                        True, image_shape, storage, verbose)
    val = _load_split(question_pipe, answer_pipe, processed_dir, splits[1],
                      False, image_shape, storage, verbose)
    return TrainingData(train, val)


def load_clevr_main(question_pipe, answer_pipe, storage='ram', image_shape=None,
                    verbose=2):
    assert storage in {'ram', 'disk'}
    image_shape = _normalize_image_shape(image_shape, _IMAGE_SHAPES)
    processed_dir = _ready(_DATASET_NAME, _MAIN_PROCESSED_SUBDIR, _MAIN_URL,
                           _MAIN_SPLITS, _IMAGE_SHAPES, verbose)
    return _load(question_pipe, answer_pipe, processed_dir, _MAIN_SPLITS,
                 image_shape, storage, verbose)


def load_clevr_cogent_same(question_pipe, answer_pipe, storage='ram',
                           image_shape=None, verbose=2):
    assert storage in {'ram', 'disk'}
    image_shape = _normalize_image_shape(image_shape, _IMAGE_SHAPES)
    processed_dir = _ready(_DATASET_NAME, _COGENT_PROCESSED_SUBDIR, _COGENT_URL,
                           _COGENT_ALL_SPLITS, _IMAGE_SHAPES, verbose)
    return _load(question_pipe, answer_pipe, processed_dir, _COGENT_SAME_SPLITS,
                 image_shape, storage, verbose)


def load_clevr_cogent_diff(question_pipe, answer_pipe, storage='ram',
                           image_shape=None, verbose=2):
    assert storage in {'ram', 'disk'}
    image_shape = _normalize_image_shape(image_shape, _IMAGE_SHAPES)
    processed_dir = _ready(_DATASET_NAME, _COGENT_PROCESSED_SUBDIR, _COGENT_URL,
                           _COGENT_ALL_SPLITS, _IMAGE_SHAPES, verbose)
    return _load(question_pipe, answer_pipe, processed_dir, _COGENT_DIFF_SPLITS,
                 image_shape, storage, verbose)
