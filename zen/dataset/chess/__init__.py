from multiprocessing import Pool
import numpy as np
import os
from zipfile import ZipFile

from ...model.data.dataset import Dataset
from ..util import download, get_dataset_dir
from .game import Game
from .pgn import PGN


_DATASET_NAME = 'chess'
_URL = 'http://kingbase-chess.net/download/484'
_URL_BASENAME = 'KingBase2017-pgn.zip'
_PROCESSED_SUBDIR = 'processed'
_PROCESSED_FILE = 'boards.npy'


def _each_pgn(text, keep_frac, verbose):
    blocks = text.strip().split('\r\n\r\n')
    assert len(blocks) % 2 == 0
    for i in range(0, len(blocks), 2):
        if keep_frac < np.random.random():
            continue
        text = '\r\n\r\n'.join([blocks[i], blocks[i + 1]])
        yield PGN.from_text(text)


def _process(zip_filename, processed_dir, keep_frac, verbose):
    print('Constructing training data from expert games (this will take ' +
          'some time...)')
    zip_ = ZipFile(zip_filename)
    paths = zip_.namelist()
    boards = []
    print('/%2d    boards  path' % len(paths))
    print('---  --------  ----')
    pool = Pool(None)
    for i, path in enumerate(paths):
        num_boards = len(boards)
        text = zip_.open(path).read().decode('latin-1')
        pgns = list(_each_pgn(text, keep_frac, verbose))
        ret = pool.map(Game.make_board_arrays, pgns)
        for list_ in ret:
            if list_ is None:
                continue
            boards += list_
        print('%3d  %8d  %s' % (i, len(boards) - num_boards, path))
    print('all  %8d  total' % len(boards))
    boards = np.array(boards)
    np.random.shuffle(boards)
    os.mkdir(processed_dir)
    out = os.path.join(processed_dir, _PROCESSED_FILE)
    np.save(out, boards)


def _ready(dataset_name, processed_subdir, url_basename, keep_frac, verbose):
    dataset_dir = get_dataset_dir(dataset_name)
    processed_dir = os.path.join(dataset_dir, processed_subdir)
    if not os.path.exists(processed_dir):
        zip_filename = os.path.join(dataset_dir, url_basename)
        if not os.path.exists(zip_filename):
            download(_URL, zip_filename, verbose)
        _process(zip_filename, processed_dir, keep_frac, verbose)
    return processed_dir


class ChessDataset(Dataset):
    def __init__(self, arrs, samples_per_epoch, val_frac, is_train):
        self.arrs = arrs
        self.samples_per_epoch = samples_per_epoch
        self.val_frac = val_frac
        self.split = int(len(self.arrs) * val_frac)
        self.is_train = is_train

    def pick_index(self):
        if self.is_train:
            index = np.random.randint(self.split, len(self.arrs))
        else:
            index = np.random.randint(0, self.split)
        return index

    def get_num_samples(self):
        return self.samples_per_epoch

    def unpack(self, index):
        arr = self.arrs[index]
        board = np.zeros((8, 8), dtype='int8')
        for y in range(8):
            row_int = arr[y]
            for x in reversed(range(8)):
                board[y][x] = row_int % 16
                row_int //= 16
        n = arr[8]
        will_win = n % 4 - 1
        n //= 4
        num_moves = n % 512
        n //= 512
        move_index = n % 512
        n //= 512
        to_x = n % 8
        n //= 8
        to_y = n % 8
        n //= 8
        from_x = n % 8
        n //= 8
        from_y = n
        to_yx = to_y, to_x
        from_yx = from_y, from_x
        return board, move_index, num_moves, will_win, from_yx, to_yx


class PieceSelectionDataset(ChessDataset):
    def get_sample(self, index):
        index = self.pick_index()
        board, move_index, num_moves, will_win, from_yx, to_yx = \
            self.unpack(index)
        board = np.equal.outer(np.arange(13), board).astype('float32')
        piece = np.zeros((64,), dtype='float32')
        piece[from_yx[0] * 8 + from_yx[1]] = 1.
        return (board,), (piece,)


class TargetSelectionDataset(ChessDataset):
    def get_sample(self, index):
        index = self.pick_index()
        board, move_index, num_moves, will_win, from_yx, to_yx = \
            self.unpack(index)
        board = np.equal.outer(np.arange(13), board).astype('float32')
        piece = np.zeros((1, 8, 8), dtype='float32')
        piece[0, from_yx[0], from_yx[1]] = 1.
        board_plus_piece = np.vstack([board, piece])
        target = np.zeros((64,), dtype='float32')
        target[to_yx[0] * 8 + to_yx[1]] = 1.
        return (board_plus_piece,), (target,)


SCALE = 1. / np.log(0 + 2)


def get_scale(moves_before):
    return 1. / np.log((moves_before / 10.) ** 3. + 2.) / SCALE


class BoardEvaluationDataset(ChessDataset):
    def get_sample(self, index):
        index = self.pick_index()
        board, move_index, num_moves, will_win, from_yx, to_yx = \
            self.unpack(index)
        board = np.equal.outer(np.arange(13), board).astype('float32')
        moves_before = num_moves - move_index
        scale = get_scale(moves_before)
        assert 0. < scale <= 1.
        happiness = will_win * scale
        happiness = np.array([happiness], dtype='float32')
        return (board,), (happiness,)


class ChessTrainingData(object):
    def __init__(self, arrs, samples_per_epoch, val_frac):
        self.arrs = arrs
        self.samples_per_epoch = samples_per_epoch
        self.val_frac = val_frac

    def for_piece_selection(self):
        return PieceSelectionDataset(self.arrs, self.samples_per_epoch,
                                     self.val_frac, True), \
            PieceSelectionDataset(self.arrs, self.samples_per_epoch,
                                  self.val_frac, False)

    def for_target_selection(self):
        return TargetSelectionDataset(self.arrs, self.samples_per_epoch,
                                      self.val_frac, True), \
            TargetSelectionDataset(self.arrs, self.samples_per_epoch,
                                   self.val_frac, False)

    def for_board_evaluation(self):
        return BoardEvaluationDataset(self.arrs, self.samples_per_epoch,
                                      self.val_frac, True), \
            BoardEvaluationDataset(self.arrs, self.samples_per_epoch,
                                   self.val_frac, False)


def load_chess(keep_frac=0.1, samples_per_epoch=100000, val_frac=0.2,
               verbose=2):
    processed_dir = _ready(_DATASET_NAME, _PROCESSED_SUBDIR, _URL_BASENAME,
                           keep_frac, verbose)
    filename = os.path.join(processed_dir, _PROCESSED_FILE)
    arrs = np.load(filename)
    return ChessTrainingData(arrs, samples_per_epoch, val_frac)
