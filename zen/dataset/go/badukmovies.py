import os
from tqdm import tqdm
from zipfile import ZipFile

from .sgf import SGF
from .util import download, get_dataset_dir


_DATASET_NAME = 'badukmovies'
_URL = 'https://badukmovies.com/pro_games/download'
_URL_BASENAME = 'badukmovies.zip'
_PROCESSED_SUBDIR = 'processed'


def _load(zip_filename, moves_basename, verbose):
    zip_ = ZipFile(zip_filename)
    infos = zip_.infolist()
    if verbose:
        print('Processing pro games from badukmovies...')
    if verbose == 2:
        infos = tqdm(infos, leave=False)
    ok = 0
    errors = []
    for info in infos:
        if not info.filename.endswith('.sgf'):
            continue
        text = zip_.open(info).read().decode('utf-8')
        is_ok = True
        try:
            yield SGF.parse(text)
        except Exception as e:
            arg, = e.args
            errors.append(arg)
            is_ok = False
        if is_ok:
            ok += 1
    if verbose:
        print('%d success, %d error (%d total).' %
              (ok, len(errors), ok + len(errors)))
    if 3 <= verbose:
        print('Errors:')
        for error in errors:
            print('* %s' % error)


def load_badukmovies(val_frac=0.2, verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    zip_filename = os.path.join(dataset_dir, _URL_BASENAME)
    if not os.path.exists(zip_filename):
        download(_URL, zip_filename, verbose)
    for sgf in _load(zip_filename, verbose):
        yield sgf
