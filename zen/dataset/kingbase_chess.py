import numpy as np
import os
from tqdm import tqdm
from zipfile import ZipFile

from ..model.data.dataset import Dataset
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

    def to_numpy(self):
        return self.arr

    @classmethod
    def from_numpy(cls, arr):
        return cls(arr)

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
        arr = np.zeros((8, 8), dtype='uint8')
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

    def find_origin_of_pawn_forward(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        if restrict_x not in {None, to_x}:
            return []

        y = to_y - 1
        x = to_x
        if self.arr[y, x] == self.my_pawn and restrict_y in {None, y}:
            return [(y, x)]
        elif to_y == 3 and self.arr[2, x] == self.space and \
                self.arr[1, x] == self.my_pawn and restrict_y in {None, 1}:
            return [(1, x)]
        else:
            return []

    def find_origin_of_pawn_en_passant_capture(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        if self.arr[to_y - 1, to_x] != self.their_pawn or \
                self.arr[to_y, to_x] != self.space:
            return []

        ret = []

        y = to_y - 1
        x = to_x - 1
        if 0 <= y < 8 and 0 <= x < 8 and self.arr[y, x] == self.my_pawn \
                and restrict_x in {None, x} and restrict_y in {None, y}:
            coords = y, x
            ret.append((coords, True))

        y = to_y - 1
        x = to_x + 1
        if 0 <= y < 8 and 0 <= x < 8 and self.arr[y, x] == self.my_pawn \
                and restrict_x in {None, x} and restrict_y in {None, y}:
            coords = y, x
            ret.append((coords, True))

        return ret

    def find_origin_of_pawn_normal_capture(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        ret = []

        y = to_y - 1
        x = to_x - 1
        if 0 <= y < 8 and 0 <= x < 8 and self.arr[y, x] == self.my_pawn \
                and restrict_x in {None, x} and restrict_y in {None, y}:
            ret.append((y, x))

        y = to_y - 1
        x = to_x + 1
        if 0 <= y < 8 and 0 <= x < 8 and self.arr[y, x] == self.my_pawn \
                and restrict_x in {None, x} and restrict_y in {None, y}:
            ret.append((y, x))

        return ret

    def find_origin_of_pawn(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to
        ret = []
        if self.arr[to] == self.space:
            ret += self.find_origin_of_pawn_forward(restrict, to)
            ret += self.find_origin_of_pawn_en_passant_capture(restrict, to)
        else:
            ret += self.find_origin_of_pawn_normal_capture(restrict, to)
        return ret

    def find_origin_of_rook(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to
        if restrict_y is not None:
            center = restrict_y, to_x
            offs = [(0, -1), (0, 1)]
        elif restrict_x is not None:
            center = to_y, restrict_x
            offs = [(-1, 0), (1, 0)]
        else:
            center = to_y, to_x
            offs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        ret = []
        for off_y, off_x in offs:
            for i in range(8):
                y = center[0] + i * off_y
                x = center[1] + i * off_x
                if y == to_y and x == to_x:
                    continue
                if not 0 <= y < 8 or not 0 <= x < 8:
                    break
                if self.arr[y, x] == self.my_rook:
                    ret.append((y, x))
                    break
                elif self.arr[y, x] != self.space:
                    break
        return list(set(ret))

    def find_origin_of_knight(self, restrict, to):
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

    def find_origin_of_bishop(self, restrict, to):
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
                    break
                n = self.arr[y, x]
                if n == self.my_bishop:
                    ret.append((y, x))

        return ret

    def find_origin_of_queen(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        ret = []

        for off_y, off_x in [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
                             (0, -1), (1, -1)]:
            for i in range(8):
                y = to_y + i * off_y
                x = to_x + i * off_x
                if y == restrict_y or x == restrict_x:
                    continue
                if not 0 <= y < 8 or not 0 <= x < 8:
                    break
                n = self.arr[y, x]
                if n == self.my_queen:
                    ret.append((y, x))

        return ret

    def find_origin_of_king(self, restrict, to):
        restrict_y, restrict_x = restrict
        to_y, to_x = to

        ret = []

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
        'p': find_origin_of_pawn,
        'r': find_origin_of_rook,
        'n': find_origin_of_knight,
        'b': find_origin_of_bishop,
        'q': find_origin_of_queen,
        'k': find_origin_of_king,
    }

    def find_origin_of(self, piece_chr, maybe_from, to):
        if maybe_from is None:
            maybe_from = None, None
        elif maybe_from[0] is not None and maybe_from[1] is not None:
            assert False
        ret = self.PIECE_CHR2FIND_ORIGIN[piece_chr](self, maybe_from, to)
        assert len(ret) == 1
        ret = ret[0]
        if isinstance(ret[1], bool):
            assert len(ret) == 2
            return ret
        else:
            return ret, False

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
        self.arr[0, 4:] = (self.space, self.my_rook, self.my_king, self.space)

    def black_kingside_castle(self):
        assert tuple(self.arr[0, :4]) == \
            (self.my_rook, self.space, self.space, self.my_king)
        self.arr[0, :4] = (self.space, self.my_king, self.my_rook,
                           self.space)

    def black_queenside_castle(self):
        assert tuple(self.arr[0, 3:]) == \
            (self.my_king, self.space, self.space, self.space, self.my_rook)
        self.arr[0, 3:] = (self.space, self.my_rook, self.my_king,
                           self.space, self.space)

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

    def move(self, from_, to, is_white, en_passant, promote_to):
        if self.arr[from_[0], from_[1]] == self.my_king and \
                abs(from_[1] - to[1]) == 2:
            self.castle(from_, to, is_white)
        else:
            if promote_to is None:
                self.arr[to[0], to[1]] = self.arr[from_[0], from_[1]]
            else:
                self.arr[to[0], to[1]] = self.chr2int[promote_to]
            self.arr[from_[0], from_[1]] = self.space
            if en_passant:
                self.arr[to[0] - 1, to[1]] = self.space

    def apply_pgn_move(self, move, is_white):
        assert move not in {'0-1', '1-0', '1/2-1/2'}
        if move == 'kingside_castle':
            if is_white:
                from_, to = (0, 4), (0, 6)
            else:
                from_, to = (0, 3), (0, 1)
            en_passant = False
            promote_to = None
        elif move == 'queenside_castle':
            if is_white:
                from_, to = (0, 4), (0, 2)
            else:
                from_, to = (0, 3), (0, 5)
            en_passant = False
            promote_to = None
        else:
            piece, maybe_from, to, capture, promote_to = move
            piece = piece.lower()
            if promote_to is not None:
                promote_to = promote_to.lower()

            if not is_white:
                maybe_from = self.rotate_coords(maybe_from)
                to = self.rotate_coords(to)

            from_, en_passant = self.find_origin_of(piece, maybe_from, to)

            piece_at_target = self.arr[to[0], to[1]]
            whose = self.whose_piece(piece_at_target)
            if capture:
                if en_passant:
                    assert whose == 'space'
                else:
                    assert whose == 'theirs'
            else:
                assert whose == 'space'

        from_pgn = self.to_pgn_coords(from_)
        to_pgn = self.to_pgn_coords(to)
        top_line = '%s %s\n' % (from_pgn, to_pgn)
        ret = top_line + self.to_text() + '\n'

        self.move(from_, to, is_white, en_passant, promote_to)

        self.rotate()

        return ret


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


def _load_board(block):
    x = block.index('\n')
    from_, to = block[:x].split()
    board = Board.from_text(block[x + 1:])
    board = board.to_numpy()
    piece = np.zeros((8, 8), dtype='uint8')
    coords = _yx_from_a1(from_)
    piece[coords] = 1
    target = np.zeros((8, 8), dtype='uint8')
    coords = _yx_from_a1(to)
    target[coords] = 1
    return board, piece, target


def _stack(tuples):
    arrs = list(zip(*tuples))
    for i, arr in enumerate(arrs):
        arrs[i] = np.stack(arr)
    return tuple(arrs)


def _load_boards(processed_dir, val_frac, verbose):
    filename = os.path.join(processed_dir, _PROCESSED_FILE)
    text = open(filename, 'r').read()
    blocks = text.split('\n\n')[:-1]
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


def _ready(dataset_name, processed_subdir, url_basename):
    dataset_dir = get_dataset_dir(dataset_name)
    processed_dir = os.path.join(dataset_dir, processed_subdir)
    if not os.path.exists(processed_dir):
        zip_filename = os.path.join(dataset_dir, url_basename)
        if not os.path.exists(zip_filename):
            download(_URL, zip_filename, verbose)
        os.mkdir(processed_dir)
        _process(zip_filename, processed_dir, verbose)
    return processed_dir


class ChessSelectPieceDataset(Dataset):
    def __init__(self, boards, pieces):
        self.boards = boards
        self.pieces = pieces

    def get_num_samples(self):
        return len(self.boards)

    def get_sample(self, index):
        indexes = self.boards[index]
        board = np.equal.outer(np.arange(13), indexes).astype('float32')
        piece = self.pieces[index].astype('float32').flatten()
        return (board,), (piece,)


class ChessSelectTargetDataset(Dataset):
    def __init__(self, boards, pieces, targets):
        self.boards = boards
        self.pieces = pieces
        self.targets = targets

    def get_num_samples(self):
        return len(self.boards)

    def get_sample(self, index):
        indexes = self.boards[index]
        board = np.equal.outer(np.arange(13), indexes).astype('float32')
        piece = self.pieces[index].astype('float32').reshape((1, 8, 8))
        board_plus_selected = np.vstack([board, piece])
        target = self.targets[index].astype('float32').flatten()
        return (board_plus_selected,), (target,)


def load_chess_moves_select_piece(val_frac=0.2, verbose=2):
    processed_dir = _ready(_DATASET_NAME, _PROCESSED_SUBDIR, _URL_BASENAME)
    train, val = _load_boards(processed_dir, val_frac, verbose)
    train_boards, train_pieces, _ = train
    train = ChessSelectPieceDataset(train_boards, train_pieces)
    val_boards, val_pieces, _ = val
    val = ChessSelectPieceDataset(val_boards, val_pieces)
    return train, val


def load_chess_moves_select_target(val_frac=0.2, verbose=2):
    processed_dir = _ready(_DATASET_NAME, _PROCESSED_SUBDIR, _URL_BASENAME)
    train, val = _load_boards(processed_dir, val_frac, verbose)
    train = ChessSelectTargetDataset(*train)
    val = ChessSelectTargetDataset(*val)
    return train, val
