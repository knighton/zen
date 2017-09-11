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


def _pgn_load(text):
    blocks = text.strip().split('\r\n\r\n')
    games = []
    for i in range(len(blocks)):
        if not i % 2:
            continue
        game = _pgn_load_moves(blocks[i])
        board = Board.initial()
        for i, move in enumerate(game[:-1]):
            is_white = not i % 2
            try:
                sample = board.apply_pgn_move(move, is_white)
            except:
                print('*' * 80)
                break
            print(sample)
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

    def flip_color(self, n):
        if not n:
            return 0
        elif 1 <= n <= 6:
            return n + 6
        elif 7 <= n <= 12:
            return n - 6
        else:
            assert False

    def rotate(self):
        self.arr = np.rot90(np.rot90(self.arr))
        for i in range(self.arr.shape[0]):
            for j in range(self.arr.shape[1]):
                self.arr[i, j] = self.flip_color(self.arr[i, j])

    def rotate_coords(self, coords):
        if coords is None:
            return None
        x, y = coords
        if x is not None:
            x = 8 - x - 1
        if y is not None:
            y = 8 - y - 1
        return x, y

    def whose_piece(self, n):
        if not n:
            return 'space'
        elif 1 <= n <= 6:
            return 'mine'
        elif 7 <= n <= 12:
            return 'theirs'
        else:
            assert False

    def find_pieces(self, n, restrict):
        yy = restrict[0] if restrict[0] is not None else \
            range(self.arr.shape[0])
        xx = restrict[1] if restrict[1] is not None else \
            range(self.arr.shape[1])
        ret = []
        for i in yy:
            for j in xx:
                if self.arr[i, j] == n:
                    ret.append((i, j))
        return ret

    def find_origin_pawn(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to
        ret = []
        if self.arr[to] == self.space:
            y = to_y - 1
            x = to_x
            if 0 <= y < 8 and 0 <= x < 8:
                if self.arr[y, x] == self.my_pawn:
                    ret.append((y, x))
                elif to_y == 3 and self.arr[y, x] == self.space:
                    y = 1
                    x = to_x
                    if 0 <= y < 8 and 0 <= x < 8:
                        if self.arr[y, x] == self.my_pawn:
                            ret.append((y, x))
        else:
            y = to_y - 1
            x = to_x - 1
            if 0 <= y < 8 and 0 <= x < 8 and self.arr[y, x] == self.my_pawn:
                ret.append((y, x))
            y = to_y - 1
            x = to_x + 1
            if 0 <= y < 8 and 0 <= x < 8 and self.arr[y, x] == self.my_pawn:
                ret.append((y, x))
        return ret

    def find_origin_rook(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        ret = []

        if restrict_y is None:
            for y in range(to_y - 1, -1, -1):
                n = self.arr[y, to_x]
                if n == self.space:
                    pass
                elif n == self.my_rook:
                    ret.append((y, to_x))
                    break
                else:
                    break

            for y in range(to_y + 1, 8):
                n = self.arr[y, to_x]
                if n == self.space:
                    pass
                elif n == self.my_rook:
                    ret.append((y, to_x))
                    break
                else:
                    break

        if restrict_x is None:
            for x in range(to_x - 1, -1, -1):
                n = self.arr[to_y, x]
                if n == self.space:
                    pass
                elif n == self.my_rook:
                    ret.append((to_y, x))
                    break
                else:
                    break

            for x in range(to_x + 1, 8):
                n = self.arr[to_y, x]
                if n == self.space:
                    pass
                elif n == self.my_rook:
                    ret.append((to_y, x))
                    break
                else:
                    break

        return ret

    def find_origin_knight(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        offsets = [
            (1, 2),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ]

        ret = []

        for off_y, off_x in offsets:
            y = to_y + off_y
            if not 0 <= y < 8:
                continue
            x = to_x + off_x
            if not 0 <= x < 8:
                continue
            n = self.arr[y, x]
            if n == self.my_knight:
                ret.append((y, x))

        if restrict_y is not None:
            ret = list(filter(lambda y_x: y_x[0] == restrict_y, ret))

        if restrict_x is not None:
            ret = list(filter(lambda y_x: y_x[1] == restrict_x, ret))

        return ret

    def find_origin_bishop(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        ret = []

        for off_y, off_x in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
            for i in range(8):
                y = to_y + i * off_y
                x = to_x + i * off_x
                if y == restrict_y or x == restrict_x:
                    continue
                if not 0 <= y < 8 or not 0 <= x < 8:
                    continue
                n = self.arr[y, x]
                if n == self.my_bishop:
                    ret.append((y, x))

        return ret

    def find_origin_queen(self, restrict, to):
        return self.find_origin_rook(restrict, to) + \
            self.find_origin_bishop(restrict, to)

    def find_origin_king(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        for off_y in [-1, 0, 1]:
            for off_x in [-1, 0, 1]:
                if off_y == off_x == 0:
                    continue
                y = to_y + off_y
                if not 0 <= y < 8:
                    continue
                x = to_x + off_x
                if not 0 <= x < 8:
                    continue
                n = self.arr[y, x]
                if n == self.my_king:
                    ret.append((y, x))

        if restrict_y is not None:
            ret = list(filter(lambda y, x: y == restrict_y, ret))

        if restrict_x is not None:
            ret = list(filter(lambda y, x: x == restrict_x, ret))

        return ret

    PIECE_CHR2FIND_ORIGIN = {
        'p': find_origin_pawn,
        'r': find_origin_rook,
        'n': find_origin_knight,
        'b': find_origin_bishop,
        'q': find_origin_queen,
        'k': find_origin_king,
    }

    def find_origin(self, piece_chr, maybe_from, to):
        if maybe_from is None:
            maybe_from = None, None
        elif maybe_from[0] is not None and maybe_from[1] is not None:
            return maybe_from
        return self.PIECE_CHR2FIND_ORIGIN[piece_chr](self, maybe_from, to)

    def to_pgn_coords(self, y_x):
        y, x = y_x
        return 'abcdefgh'[x] + '12345678'[y]

    def white_queenside_castle(self):
        assert tuple(self.arr[0, :5]) == \
            (self.my_rook, self.space, self.space, self.space, self.my_king)
        self.arr[0, :5] = (self.space, self.space, self.my_king,
                           self.my_rook, self.space)

    def white_kingside_castle(self):
        assert tuple(self.arr[0, 4:]) == \
            (self.my_king, self.space, self.space, self.my_rook)
        self.arr[0, 4:] == (self.space, self.my_rook, self.my_king,
                            self.space)

    def black_kingside_castle(self):
        assert tuple(self.arr[0, :4]) == \
            (self.my_rook, self.space, self.space, self.my_king)
        self.arr[0, :4] = (self.space, self.my_rook, self.my_king,
                           self.space)

    def black_queenside_castle(self):
        assert tuple(self.arr[0, 3:]) == \
            (self.my_king, self.space, self.space, self.space, self.my_rook)
        self.arr[0, 3:] = (self.space, self.my_rook, self.my_king,
                           self.space)

    def castle(self, from_, to, is_white):
        if is_white:
            assert from_ == (0, 4)
            if to == (0, 2):
                self.white_queenside_castle()
            elif to == (0, 6):
                self.white_kingside_castle()
            else:
                assert False
        else:
            assert from_ == (0, 3)
            if to == (0, 1):
                self.black_kingside_castle()
            elif to == (0, 5):
                self.black_queenside_castle()
            else:
                assert False

    def move(self, from_, to, is_white):
        if self.arr[from_[0], from_[1]] == self.my_king and \
                abs(from_[1] - to[1]) == 2:
            self.castle(from_, to, is_white)
        else:
            self.arr[to[0], to[1]] = self.arr[from_[0], from_[1]]
            self.arr[from_[0], from_[1]] = self.space

    def apply_pgn_move(self, move, is_white):
        assert move not in {'0-1', '1-0', '1/2-1/2'}
        if move == 'kingside_castle':
            if is_white:
                from_, to = (0, 4), (0, 6)
            else:
                from_, to = (0, 3), (0, 1)
        elif move == 'queenside_castle':
            if is_white:
                from_, to = (0, 4), (0, 2)
            else:
                from_, to = (0, 3), (0, 5)
        else:
            piece, maybe_from, to, capture, promote_to = move
            piece = piece.lower()

            if promote_to is not None:
                raise NotImplementedError  # XXX

            if not is_white:
                maybe_from = self.rotate_coords(maybe_from)
                to = self.rotate_coords(to)

            piece_at_target = self.arr[to[0], to[1]]
            whose = self.whose_piece(piece_at_target)
            if capture:
                assert whose == 'theirs'
            else:
                assert whose == 'space'

            froms = self.find_origin(piece, maybe_from, to)
            assert len(froms) == 1, str(froms)
            from_ = froms[0]

        from_pgn = self.to_pgn_coords(from_)
        to_pgn = self.to_pgn_coords(to)
        top_line = '%s %s\n' % (from_pgn, to_pgn)
        ret = top_line + self.to_text()

        self.move(from_, to, is_white)

        self.rotate()

        return ret


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
