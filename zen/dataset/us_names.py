import os
from tqdm import tqdm
from zipfile import ZipFile

from .util import download, get_dataset_dir


_DATASET_NAME = 'us_names'
_URL = 'https://www.ssa.gov/oact/babynames/state/namesbystate.zip'
_ENCODINGS = ['utf-8', 'latin-1']


def _load(zip_filename, verbose):
    zip_file = ZipFile(zip_filename)
    infos = zip_file.infolist()
    if verbose == 2:
        infos = tqdm(infos)
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
    aaa.sort()
    states, genders, years, names, counts = zip(*aaa)
    print(len(set(names)))
    print(sum(counts))
    return aaa


def load_us_names(verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    zip_filename = os.path.join(dataset_dir, os.path.basename(_URL))
    download(_URL, zip_filename, verbose)
    return _load(zip_filename, verbose)
