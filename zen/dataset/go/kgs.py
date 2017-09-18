from glob import glob
import os
from tqdm import tqdm
from zipfile import ZipFile

from ..util import download, get_dataset_dir
from .sgf import SGF


_DATASET_NAME = 'kgs'
_URL = 'https://www.u-go.net/gamerecords/'


def _get_zip_urls(index_filename):
    text = open(index_filename).read()
    ss = text.split('https://')[1:]
    urls = map(lambda s: 'https://' + s[:s.index('"')], ss)
    zips = filter(lambda s: s.endswith('.zip'), urls)
    return sorted(zips)


def _each_sgf_from_zip(zip_filename, verbose, filename_index, num_filenames):
    zip_ = ZipFile(zip_filename)
    infos = zip_.infolist()
    if verbose:
        print('%2d  %5d  %s' %
              (filename_index, len(infos),
               os.path.basename(zip_filename)))
    if verbose == 2:
        infos = tqdm(infos, leave=False)
    for info in infos:
        if not info.filename.endswith('.sgf'):
            continue
        text = zip_.open(info).read().decode('utf-8')
        yield SGF.load(text)


def _load(dataset_dir, verbose):
    zip_filenames = glob(os.path.join(dataset_dir, '*.zip'))
    zip_filenames.sort()
    z = len(zip_filenames)
    if verbose:
        print('Processing one 7d+/both 6d+ KGS baduk games...')
        print('%2d  count  filename' % (z,))
        print('--  -----  --------')
    for i, zip_filename in enumerate(zip_filenames):
        for sgf in _each_sgf_from_zip(zip_filename, verbose, i, z):
            yield sgf


def load_kgs(verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    index_filename = os.path.join(dataset_dir, 'index.html')
    if not os.path.exists(index_filename):
        download(_URL, index_filename, verbose)
    zip_urls = _get_zip_urls(index_filename)
    for zip_url in zip_urls:
        zip_filename = os.path.join(dataset_dir, os.path.basename(zip_url))
        if not os.path.exists(zip_filename):
            download(zip_url, zip_filename, 1)
    for sgf in _load(dataset_dir, verbose):
        yield sgf
