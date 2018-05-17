import numpy as np
import os
from tqdm import tqdm
from zipfile import ZipFile

from ..model.data.dataset import Dataset
from .chessboard import Board
from .util import download, get_dataset_dir


_DATASET_NAME = 'kingbase_chess'
_URL = 'http://kingbase-chess.net/download/484'
_URL_BASENAME = 'KingBase2017-pgn.zip'
_PROCESSED_SUBDIR = 'processed'
_PROCESSED_FILE = 'boards.txt'


def _pgn_load_without_comments(text):
    while '{' in text:
        a = text.find('{')
        z = text.find('}')
        assert a < z
        text = text[:a] + text[z + 1:]
    return text


def _pgn_load_move_to(text):
    assert len(text) == 2
    a, b = text
    assert 'a' <= text[0] <= 'h'
    assert '1' <= text[1] <= '8'
    x = ord(text[0]) - ord('a')
    y = ord(text[1]) - ord('1')
    return y, x


def _pgn_load_move(text):
    if text.endswith('+') or text.endswith('#'):
        text = text[:-1]

    if text in {'0-1', '1-0', '1/2-1/2', '*'}:
        return text

    if text == 'O-O':
        return 'kingside_castle'

    if text == 'O-O-O':
        return 'queenside_castle'

    if 'x' in text:
        capture = True
        text = text.replace('x', '')
    else:
        capture = False

    if '=' in text:
        eq = text.find('=')
        promote_to = text[eq + 1:]
        text = text[:eq]
    else:
        promote_to = None

    try:
        move_to = _pgn_load_move_to(text[-2:])
    except:
        raise

    c = text[:-2]
    move_from = None
    if not len(c):
        piece = 'P'
    elif len(c) == 1:
        if c.isupper():
            piece = c
        elif c.islower():
            piece = 'P'
            x = ord(c) - ord('a')
            move_from = None, x
        else:
            assert False
    elif len(c) == 2:
        if c[0].isupper() and c[1].islower():
            piece = c[0]
            x = ord(c[1]) - ord('a')
            move_from = None, x
        elif c[0].isupper() and c[1].isdigit():
            piece = c[0]
            y = ord(c[1]) - ord('1')
            move_from = y, None
        else:
            assert False
    else:
        assert False

    return piece, move_from, move_to, capture, promote_to


def _pgn_load_moves(text):
    text = _pgn_load_without_comments(text)
    if '...' in text:
        return None
    text = text.replace('.', ' . ')
    ss = text.split()
    if not len(ss) % 4:
        pass
    elif len(ss) % 4 == 1:
        assert ss[-1] in {'0-1', '1-0', '1/2-1/2', '*'}, text
    else:
        assert False, text
    moves = []
    for i in range(len(ss) // 4):
        assert int(ss[i * 4]) == i + 1
        assert ss[i * 4 + 1] == '.'
        white = _pgn_load_move(ss[i * 4 + 2])
        if white == '*':
            return None
        moves.append(white)
        black = _pgn_load_move(ss[i * 4 + 3])
        if black == '*':
            return None
        moves.append(black)
    return moves


def _process_pgns(text):
    blocks = text.strip().split('\r\n\r\n')
    samples = []
    for i in range(len(blocks)):
        if not i % 2:
            continue
        if 0.01 < np.random.random():
            continue
        game = _pgn_load_moves(blocks[i])
        if game is None:
            continue
        board = Board.initial()
        for j, move in enumerate(game[:-1]):
            is_white = not j % 2
            try:
                sample = board.apply_pgn_move(move, is_white)
            except:
                print('Ambiguity at game %d, move %d, gave up.' % (i, j))
                break
            samples.append(sample)
    return samples


def _process(zip_filename, processed_dir, verbose):
    zip_ = ZipFile(zip_filename)
    paths = zip_.namelist()
    if verbose == 2:
        paths = tqdm(paths, leave=False)
    samples = []
    for path in paths:
        if np.random.uniform() < 0.9:
            continue
        text = zip_.open(path).read().decode('latin-1')
        samples += _process_pgns(text)
    np.random.shuffle(samples)
    out = os.path.join(processed_dir, _PROCESSED_FILE)
    with open(out, 'w') as out:
        for sample in samples:
            out.write(sample)


def _yx_from_a1(a1):
    x = ord(a1[0]) - ord('a')
    y = ord(a1[1]) - ord('1')
    return y, x


def a1_from_yx(yx):
    y, x = yx
    _18 = '12345678'[y]
    ah = 'abcdefgh'[x]
    return ah + _18


def _load_board(block):
    x = block.index('\n')
    from_, to = block[:x].split()
    board = Board.from_text(block[x + 1:])
    board = board.to_numpy()
    yx = _yx_from_a1(from_)
    index = yx[0] * 8 + yx[1]
    from_ = np.array([index], dtype='int8')
    yx = _yx_from_a1(to)
    index = yx[0] * 8 + yx[1]
    to = np.array([index], dtype='int8')
    return board, from_, to


def _stack(tuples):
    arrs = list(zip(*tuples))
    for i, arr in enumerate(arrs):
        arrs[i] = np.stack(arr)
    return tuple(arrs)


def _load_boards(processed_dir, val_frac, verbose):
    filename = os.path.join(processed_dir, _PROCESSED_FILE)
    text = open(filename, 'r').read()
    blocks = text.split('\n\n')[:-1][:1000000]
    if verbose == 2:
        blocks = tqdm(blocks, leave=False)
    tuples = []
    for block in blocks:
        arrs = _load_board(block)
        tuples.append(arrs)
    split = int(len(tuples) * val_frac)
    train = _stack(tuples[split:])
    val = _stack(tuples[:split])
    return train, val


def _ready(dataset_name, processed_subdir, url_basename, verbose):
    dataset_dir = get_dataset_dir(dataset_name)
    processed_dir = os.path.join(dataset_dir, processed_subdir)
    if not os.path.exists(processed_dir):
        zip_filename = os.path.join(dataset_dir, url_basename)
        if not os.path.exists(zip_filename):
            download(_URL, zip_filename, verbose)
        os.mkdir(processed_dir)
        _process(zip_filename, processed_dir, verbose)
    return processed_dir


class ChessPieceSelectionDataset(Dataset):
    def __init__(self, boards, froms):
        self.boards = boards
        self.froms = froms

    def get_num_samples(self):
        return len(self.boards)

    def get_sample(self, index):
        indexes = self.boards[index]
        board = np.equal.outer(np.arange(13), indexes).astype('float32')
        from_ = np.zeros((64,), dtype='float32')
        from_[self.froms[index]] = 1.
        return (board,), (from_,)


class ChessTargetSelectionDataset(Dataset):
    def __init__(self, boards, froms, tos):
        self.boards = boards
        self.froms = froms
        self.tos = tos

    def get_num_samples(self):
        return len(self.boards)

    def get_sample(self, index):
        indexes = self.boards[index]
        board = np.equal.outer(np.arange(13), indexes).astype('float32')
        from_ = np.zeros((64,), dtype='float32')
        from_[self.froms[index]] = 1.
        from_ = from_.reshape((1, 8, 8))
        board_plus_from = np.vstack([board, from_])
        to = np.zeros((64,), dtype='float32')
        to[self.tos[index]] = 1.
        return (board_plus_from,), (to,)


def load_chess_piece_selection(val_frac=0.2, verbose=2):
    processed_dir = _ready(_DATASET_NAME, _PROCESSED_SUBDIR, _URL_BASENAME,
                           verbose)
    train, val = _load_boards(processed_dir, val_frac, verbose)
    train_boards, train_pieces, _ = train
    train = ChessPieceSelectionDataset(train_boards, train_pieces)
    val_boards, val_pieces, _ = val
    val = ChessPieceSelectionDataset(val_boards, val_pieces)
    return train, val


def load_chess_target_selection(val_frac=0.2, verbose=2):
    processed_dir = _ready(_DATASET_NAME, _PROCESSED_SUBDIR, _URL_BASENAME,
                           verbose)
    train, val = _load_boards(processed_dir, val_frac, verbose)
    train = ChessTargetSelectionDataset(*train)
    val = ChessTargetSelectionDataset(*val)
    return train, val
