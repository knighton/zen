import numpy as np
import os
from tqdm import tqdm
from zipfile import ZipFile

from .util import download, get_dataset_dir


_DATASET_NAME = 'kingbase_chess'
_URL = 'http://kingbase-chess.net/download/484'
_URL_BASENAME = 'KingBase2017-pgn.zip'
_PROCESSED_SUBDIR = 'processed'


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
    return ord(text[0]) - ord('a'), ord(text[1]) - ord('1')


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
        text = text[:eq]
        promote_to = text[eq + 1:]
    else:
        promote_to = None

    try:
        move_to = _pgn_load_move_to(text[-2:])
    except:
        print(text)
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
            move_from = ord(c) - ord('a'), None
        else:
            assert False
    elif len(c) == 2:
        if c[0].isupper() and c[1].islower():
            piece = c[0]
            move_from = ord(c[1]) - ord('a'), None
        elif c[0].isupper() and c[1].isdigit():
            piece = c[0]
            move_from = None, ord(c[1]) - ord('1')
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


def _pgn_load(text):
    blocks = text.strip().split('\r\n\r\n')
    games = []
    for i in range(len(blocks)):
        if not i % 2:
            continue
        game = _pgn_load_moves(blocks[i])
        board = Board.initial()
        for i, move in enumerate(game):
            is_white = not i % 2
            board.apply_pgn_move(move, is_white)
    return games


class Board(object):
    int2chr = '.prnbqkPRNBQK'

    chr2int = {}
    for i, c in enumerate(int2chr):
        chr2int[c] = i

    space, my_pawn, my_rook, my_knight, my_bishop, my_queen, my_king, \
    their_pawn, their_rook, their_knight, their_bishop, their_queen, \
    their_king = range(13)

    def __init__(self, arr):
        self.arr = arr

    def to_text(self):
        lines = []
        for y in reversed(range(8)):
            line = []
            for x in range(8):
                n = self.arr[y, x]
                c = self.int2chr[n]
                line.append(c)
            lines.append(''.join(line))
        return ''.join(map(lambda line: line + '\n', lines))

    @classmethod
    def from_text(cls, text):
        lines = text.strip().split()
        assert len(lines) == 8
        arr = np.zeros((8, 8), dtype='int8')
        for y, line in enumerate(lines):
            assert len(line) == 8
            for x, c in enumerate(line):
                arr[8 - y - 1, x] = cls.chr2int[c]
        return cls(arr)

    @classmethod
    def initial(cls):
        return cls.from_text("""
            RNBQKBNR
            PPPPPPPP
            ........
            ........
            ........
            ........
            pppppppp
            rnbqkbnr
        """)

    def apply_pgn_move(self, move, is_white):
        print('apply_pgn_move', move, is_white)


def _process(zip_filename, processed_dir, verbose):
    zip_ = ZipFile(zip_filename)
    paths = zip_.namelist()
    if verbose == 2:
        paths = tqdm(paths, leave=False)
    for path in paths:
        text = zip_.open(path).read().decode('latin-1')
        games = _pgn_load(text)


def _load(processed_dir, val_frac, verbose):
    raise NotImplementedError # XXX


def load_kingbase_chess(val_frac=0.2, verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    processed_dir = os.path.join(dataset_dir, _PROCESSED_SUBDIR)
    if not os.path.exists(processed_dir):
        zip_filename = os.path.join(dataset_dir, _URL_BASENAME)
        download(_URL, zip_filename, verbose)
        _process(zip_filename, processed_dir, verbose)
    return _load(processed_dir, val_frac, verbose)
