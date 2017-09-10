from bisect import bisect_right
from collections import Counter
import numpy as np
import os
from tqdm import tqdm
from zipfile import ZipFile

from ..model.data.dataset import Dataset
from ..model.data.training_data import TrainingData
from .util import download, get_dataset_dir


_DATASET_NAME = 'us_names'
_URL = 'https://www.ssa.gov/oact/babynames/state/namesbystate.zip'
_ENCODINGS = ['utf-8', 'latin-1']


def _load(zip_filename, verbose):
    zip_file = ZipFile(zip_filename)
    infos = zip_file.infolist()
    if verbose == 2:
        infos = tqdm(infos, leave=False)
    aaa = []
    for info in infos:
        if not info.filename.endswith('.TXT'):
            continue
        for line in zip_file.open(info).readlines():
            for encoding in _ENCODINGS:
                try:
                    line = line.decode(encoding).strip()
                    break
                except:
                    pass
            state, gender, year, name, count = line.split(',')
            year = int(year)
            count = int(count)
            aa = state, gender, year, name, count
            aaa.append(aa)
    return aaa


def load_us_names(verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    zip_filename = os.path.join(dataset_dir, os.path.basename(_URL))
    download(_URL, zip_filename, verbose)
    return _load(zip_filename, verbose)


class USNameGenderDataset(Dataset):
    """
    Samples Americans.
    """

    def __init__(self, segments, total_population, max_name_len=16,
                 samples_per_epoch=64 * 1024):
        self.names, self.genders, self.total_populationes = zip(*segments)
        self.total_population = total_population
        self.max_name_len = max_name_len
        self.samples_per_epoch = samples_per_epoch

    def get_num_samples(self):
        return self.samples_per_epoch

    def name_to_tokens(self, name):
        nn = list(map(ord, name))
        nn = list(filter(lambda n: n if n < 128 else 0, nn))
        nn = nn[:self.max_name_len]
        z = self.max_name_len - len(nn)
        return np.array(nn + [0] * z).astype('int64')

    def gender_to_float(self, gender):
        value = {
            'F': 0.,
            'M': 1.,
        }[gender]
        return np.array([value]).astype('float32')

    def get_sample(self, _):
        person_index = np.random.randint(0, self.total_population)
        segment_index = bisect_right(self.total_populationes, person_index)
        name = self.names[segment_index]
        name = self.name_to_tokens(name)
        gender = self.genders[segment_index]
        gender = self.gender_to_float(gender)
        return (name,), (gender,)


def load_us_name_gender(max_name_len=16, samples_per_epoch=64 * 1024,
                        verbose=2):
    tuples = load_us_names(verbose)
    name_gender2count = Counter()
    tuples = tqdm(tuples, leave=False)
    for state, gender, year, name, count in tuples:
        name_gender2count[(name, gender)] += count
    segments = []
    total_population = 0
    for name, gender in sorted(name_gender2count):
        segment = name, gender, total_population
        segments.append(segment)
        total_population += name_gender2count[(name, gender)]
    dataset = USNameGenderDataset(segments, total_population, max_name_len,
                                  samples_per_epoch)
    return TrainingData(dataset, dataset)
