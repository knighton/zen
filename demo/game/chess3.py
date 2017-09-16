from argparse import ArgumentParser
from colorama import Back, Fore, Style
import chess.uci
import numpy as np

from zen.dataset.chess import load_chess
from zen.layer import *  # noqa


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--keep_frac', type=float, default=0.1)
    ap.add_argument('--samples_per_epoch', type=int, default=10000)
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--load_verbose', type=int, default=2)
    ap.add_argument('--train_piece_stop', type=int, default=1)
    ap.add_argument('--train_target_stop', type=int, default=1)
    ap.add_argument('--train_board_stop', type=int, default=1)
    ap.add_argument('--train_verbose', type=int, default=2)
    ap.add_argument('--play', type=str, default='console')
    ap.add_argument('--engine_binary', type=str,
                    default='/usr/local/bin/stockfish_8_x64')
    ap.add_argument('--time_per_move', type=float, default=0.01)
    return ap.parse_args()


conv = lambda n: Conv(n) > BatchNorm > ReLU > Z
cnn = lambda n: conv(n) > conv(n) > conv(n) > conv(n) > Z


def get_piece_model():
    return Input((13, 8, 8)) > cnn(128) > Conv(1) > Flatten > Softmax > Z


def get_target_model():
    return Input((14, 8, 8)) > cnn(128) > Conv(1) > Flatten > Softmax > Z


def get_win_model():
    return Input((13, 8, 8)) > cnn(128) > conv(1) > Flatten > Dense(1) > \
        Tanh > Z


class Game(object):
    def __init__(self, process):
        self.process = process
        self.board = chess.Board()

    def move(self, move):
        assert self.board.is_legal(move)
        self.board.push(move)
        self.process.position(self.board)


def get_piece_legality_mask(game):
    mask = np.zeros((64,), dtype=bool)
    for move in game.board.legal_moves:
        mask[move.from_square] = True
    return mask.reshape((8, 8))


def get_target_legality_mask(game, from_square):
    mask = np.zeros((64,), dtype=bool)
    for move in game.board.legal_moves:
        if move.from_square != from_square:
            continue
        mask[move.to_square] = True
    return mask.reshape((8, 8))


INT2CHR = '.PRNBQKprnbqk'


CHR2INT = {}
for i, c in enumerate(INT2CHR):
    CHR2INT[c] = i


def text_to_grid(text):
    cc = text.split()
    grid = np.zeros((8, 8), dtype='int8')
    for i, c in enumerate(cc):
        grid[i // 8, i % 8] = CHR2INT[c]
    return np.flip(grid, 0)


def select(heatmap, strategy):
    if strategy == 'best':
        index = heatmap.argmax()
    elif strategy == 'sample':
        cumsum = heatmap.cumsum()
        rand = np.random.random()
        for index, cumsum_up_to in enumerate(cumsum):
            if rand < cumsum_up_to:
                break
    else:
        assert False, 'Unknown strategy (%s).' % strategy
    return index


def input_with_default(text, default):
    s = input(text)
    if not s:
        s = default
    return s


def heat_to_color(heat):
    if heat is None:
        color = Fore.WHITE
    elif heat < 0.001:
        color = Style.DIM + Fore.BLUE
    elif heat <= 0.01:
        color = Fore.BLUE
    elif heat < 0.025:
        color = Style.BRIGHT + Fore.BLUE
    elif heat < 0.1:
        color = Fore.CYAN
    elif heat < 0.2:
        color = Style.BRIGHT + Fore.GREEN
    elif heat < 0.5:
        color = Style.BRIGHT + Fore.YELLOW
    else:
        color = Style.BRIGHT + Fore.RED
    return color


def dump_pretty_board(board, heatmap, selected_square):
    if selected_square is None:
        selected_yx = None
    else:
        selected_yx = selected_square // 8, selected_square % 8
    lines = []
    lines.append(Fore.WHITE + Style.DIM + '  ╔' + '─' * 17 + '╗' +
                 Style.RESET_ALL)
    for y in reversed(range(8)):
        if selected_yx is not None and y == selected_yx[0]:
            color = Fore.WHITE + Style.BRIGHT
        else:
            color = Fore.WHITE + Style.DIM
        line = [color + str(y + 1), '│'+ Style.RESET_ALL]
        for x in range(8):
            n = board[y, x]
            c = INT2CHR[n]
            heat = None if heatmap is None else heatmap[y][x]
            if c == '.':
                c = '.' if heat is None or heat < 0.001 else '■'
            if selected_yx is not None and selected_yx == (y, x):
                color = Fore.BLACK + Back.WHITE
            else:
                color = heat_to_color(heat)
            line.append(color + c + Style.RESET_ALL)
        line.append(Fore.WHITE + Style.DIM + '│' + Style.RESET_ALL)
        lines.append(' '.join(line))
    left = Fore.WHITE + Style.DIM + '  ╚─' + Style.RESET_ALL
    middle = []
    for x in range(8):
        if selected_yx is not None and selected_yx[1] == x:
            color = Fore.WHITE + Style.BRIGHT
        else:
            color = Fore.WHITE + Style.DIM
        middle.append(color + '─' + Style.RESET_ALL)
    middle = (Fore.WHITE + Style.DIM + '─' + Style.RESET_ALL).join(middle)
    right = Fore.WHITE + Style.DIM + '─╝' + Style.RESET_ALL
    lines.append(left + middle + right)
    line = []
    for i, c in enumerate('abcdefgh'):
        if selected_yx is not None and i == selected_yx[1]:
            color = Fore.WHITE + Style.BRIGHT
        else:
            color = Fore.WHITE + Style.DIM
        line.append(color + c + Style.RESET_ALL)
    lines.append('    ' + ' '.join(line))
    text = '\n'.join(map(lambda line: ' ' * 4 + line, lines))
    print(text)


def square_to_a1(index):
    y = '12345678'[index // 8]
    x = 'abcdefgh'[index % 8]
    return x + y


def a1_to_square(a1):
    x, y = a1
    assert 'a' <= x <= 'h'
    x = ord(x) - ord('a')
    assert '1' <= y <= '8'
    y = ord(y) - ord('1')
    return y * 8 + x


def switch_player(n):
    if n == 0:
        return n
    elif 1 <= n < 7:
        return n + 6
    elif 7 <= n < 13:
        return n - 6
    else:
        assert False


def rotate_square(index):
    y, x = index // 8, index % 8
    y = 7 - y
    x = 7 - x
    return y * 8 + x


def get_from_square(piece_model, game, is_human, rotate, strategy='best',
                    their_to_square=None):
    grid = text_to_grid(str(game.board))
    if rotate:
        grid = np.rot90(np.rot90(grid))
        for y in range(8):
            for x in range(8):
                grid[y, x] = switch_player(grid[y, x])
    board = np.equal.outer(np.arange(13), grid).astype('float32')
    board = np.expand_dims(board, 0)
    is_legal = get_piece_legality_mask(game)
    assert is_legal.sum()
    heatmap, = piece_model.predict_on_batch([board])
    if rotate:
        heatmap = np.rot90(np.rot90(heatmap))
    heatmap = heatmap.reshape((8, 8))
    heatmap = heatmap * is_legal
    heatmap /= heatmap.sum()
    from_square = select(heatmap, strategy)
    if is_human:
        dump_pretty_board(grid, heatmap, their_to_square)
        print()
        while True:
            from_a1 = square_to_a1(from_square)
            from_a1 = input_with_default(
                ' ' * 8 + 'Piece (%s): ' % from_a1, from_a1)
            try:
                from_square = a1_to_square(from_a1)
            except:
                print('Not a square.')
                continue
            if not is_legal[from_square // 8, from_square % 8]:
                print('Illegal move.')
                continue
            break
        print()
        print()
    return from_square, heatmap


def get_to_square(target_model, game, from_square, is_human, rotate,
                  strategy='best'):
    grid = text_to_grid(str(game.board))
    if rotate:
        grid = np.rot90(np.rot90(grid))
        for y in range(8):
            for x in range(8):
                grid[y, x] = switch_player(grid[y, x])
        from_square = rotate_square(from_square)
    board = np.equal.outer(np.arange(13), grid).astype('float32')
    board = np.expand_dims(board, 0)
    from_ = np.zeros((1, 1, 8, 8), dtype='float32')
    from_[0, 0, from_square // 8, from_square % 8] = 1.
    board_cat_from = np.concatenate([board, from_], 1)
    if rotate:
        from_square = rotate_square(from_square)
    is_legal = get_target_legality_mask(game, from_square)
    assert is_legal.sum()
    heatmap, = target_model.predict_on_batch([board_cat_from])
    if rotate:
        heatmap = np.rot90(np.rot90(heatmap))
    heatmap = heatmap.reshape((8, 8))
    heatmap = heatmap * is_legal
    heatmap /= heatmap.sum()
    to_square = select(heatmap, strategy)
    if is_human:
        dump_pretty_board(grid, heatmap, from_square)
        print()
        while True:
            to_a1 = square_to_a1(to_square)
            to_a1 = input_with_default(
                ' ' * 8 + 'Target (%s): ' % to_a1, to_a1)
            try:
                to_square = a1_to_square(to_a1)
            except:
                print('Not a square.')
                continue
            if not is_legal[to_square // 8, to_square % 8]:
                print('Illegal move.')
                continue
            break
        print()
        print()
    return to_square


def console(process, time_per_move, piece_model, target_model, win_model,
            my_strategy='best'):
    game = Game(process)
    their_to_square = None
    while True:
        from_square, from_heatmap = get_from_square(
            piece_model, game, True, False, 'best', their_to_square)
        to_square = get_to_square(
            target_model, game, from_square, True, False, 'best')
        move = chess.Move(from_square, to_square)
        game.move(move)

        grid = text_to_grid(str(game.board))
        dump_pretty_board(grid, from_heatmap, to_square)
        print()
        print(' ' * 8 + 'Your move')
        print()

        from_square, from_heatmap = get_from_square(
            piece_model, game, False, True, 'best', None)
        their_to_square = get_to_square(
            target_model, game, from_square, False, True, 'best')
        move = chess.Move(from_square, their_to_square)
        game.move(move)

        grid = text_to_grid(str(game.board))
        dump_pretty_board(grid, from_heatmap, their_to_square)
        print()
        print(' ' * 8 + 'Their move')
        print()
        print()
        print()


def run(args):
    print('Hooking up to the chess engine...')
    process = chess.uci.popen_engine(args.engine_binary)

    print('Building models...')
    piece_model = get_piece_model()
    target_model = get_target_model()
    win_model = get_win_model()

    print('Loading data...')
    data = load_chess(args.keep_frac, args.samples_per_epoch, args.val_frac,
                      args.load_verbose)

    print('Training piece selection model...')
    piece_model.train_classifier(
        data.for_piece_selection(), stop=args.train_piece_stop,
        verbose=args.train_verbose)

    print('Training target selection model...')
    target_model.train_classifier(
        data.for_target_selection(), stop=args.train_target_stop,
        verbose=args.train_verbose)

    print('Training board evaluation model...')
    win_model.train_regressor(
        data.for_board_evaluation(), stop=args.train_board_stop,
        verbose=args.train_verbose)

    if args.play == 'console':
        print('Starting game...')
        console(process, args.tie_per_move, piece_model, target_model,
                win_model)


if __name__ == '__main__':
    run(parse_args())
