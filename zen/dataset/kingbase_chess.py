import os
from tqdm import tqdm
from zipfile import ZipFile

from .util import download, get_dataset_dir


_DATASET_NAME = 'kingbase_chess'
_URL = 'http://kingbase-chess.net/download/484'
_URL_BASENAME = 'KingBase2017-pgn.zip'


def _without_comments(text):
    while '{' in text:
        a = text.find('{')
        z = text.find('}')
        assert a < z
        text = text[:a] + text[z + 1:]
    return text


def _load_move_to(text):
    assert len(text) == 2
    a, b = text
    assert 'a' <= text[0] <= 'h'
    assert '1' <= text[1] <= '8'
    return ord(text[0]) - ord('a'), ord(text[1]) - ord('1')


def _load_move(text):
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
        move_to = _load_move_to(text[-2:])
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


def _load_moves(text):
    text = _without_comments(text)
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
        white = _load_move(ss[i * 4 + 2])
        if white == '*':
            return None
        moves.append(white)
        black = _load_move(ss[i * 4 + 3])
        if black == '*':
            return None
        moves.append(black)
    return moves


def _load(filename, val_frac, verbose):
    zip_ = ZipFile(filename)
    paths = zip_.namelist()
    if verbose == 2:
        paths = tqdm(paths, leave=False)
    games = []
    for path in paths:
        text = zip_.open(path).read().decode('latin-1')
        blocks = text.strip().split('\r\n\r\n')
        for i in range(len(blocks)):
            if not i % 2:
                continue
            game = _load_moves(blocks[i])


def load_kingbase_chess(val_frac=0.2, verbose=2):
    dataset_dir = get_dataset_dir(_DATASET_NAME)
    local = os.path.join(dataset_dir, _URL_BASENAME)
    download(_URL, local, verbose)
    return _load(local, val_frac, verbose)
