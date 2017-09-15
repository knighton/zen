import numpy as np
import os
from tqdm import tqdm
from zipfile import ZipFile

from .game import Game
from .pgn import PGN
from ..util import download, get_dataset_dir


_DATASET_NAME = 'chess'
_URL = 'http://kingbase-chess.net/download/484'
_URL_BASENAME = 'KingBase2017-pgn.zip'
_PROCESSED_SUBDIR = 'processed'
_PROCESSED_FILE = 'boards.txt'


def _each_pgn(text):
    blocks = text.strip().split('\r\n\r\n')
    assert len(blocks) % 2 == 0
    for i in range(0, len(blocks), 2):
        if 0.001 < np.random.random():
            continue
        text = '\r\n\r\n'.join([blocks[i], blocks[i + 1]])
        yield PGN.from_text(text)


def _process(zip_filename, processed_dir, verbose):
    zip_ = ZipFile(zip_filename)
    paths = zip_.namelist()
    if verbose == 2:
        paths = tqdm(paths, leave=False)
    boards = []
    for path in paths:
        text = zip_.open(path).read().decode('latin-1')
        for pgn in _each_pgn(text):
            for board in Game.each_board(pgn):
                boards.append(board)
    np.random.shuffle(boards)
    os.mkdir(processed_dir)
    out = os.path.join(processed_dir, _PROCESSED_FILE)
    with open(out, 'w') as out:
        for board in boards:
            out.write(board)


def _ready(dataset_name, processed_subdir, url_basename, verbose):
    dataset_dir = get_dataset_dir(dataset_name)
    processed_dir = os.path.join(dataset_dir, processed_subdir)
    if not os.path.exists(processed_dir):
        zip_filename = os.path.join(dataset_dir, url_basename)
        if not os.path.exists(zip_filename):
            download(_URL, zip_filename, verbose)
        _process(zip_filename, processed_dir, verbose)
    return processed_dir


def load_chess(val_frac=0.2, verbose=2):
    processed_dir = _ready(_DATASET_NAME, _PROCESSED_SUBDIR, _URL_BASENAME,
                           verbose)
