from argparse import ArgumentParser
import chess.uci
import numpy as np

from zen.layer import *  # noqa


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--engine_binary', type=str,
                    default='/usr/local/bin/stockfish_8_x64')
    ap.add_argument('--time_per_move', type=float, default=0.01)
    return ap.parse_args()


class ChessEngine(object):
    def __init__(self, process):
        self.process = process

    def move(self, board, move):
        assert board.is_legal(move)
        board.push(move)
        self.process.position(board)


class Player(object):
    def get_move(self, board):
        raise NotImplementedError


class ComputerPlayer(Player):
    def __init__(self, process, time_per_move):
        self.process = process
        self.time_per_move = time_per_move

    def get_move(self, board):
        return self.process.go(movetime=self.time_per_move).bestmove


def play_a_game(process, time_per_move):
    engine = ChessEngine(process)
    white = black = ComputerPlayer(process, time_per_move)
    board = chess.Board()
    process.position(board)
    boards = []
    moves = []
    while True:
        move = white.get_move(board)
        boards.append(str(board))
        moves.append(move)
        engine.move(board, move)
        if board.is_game_over():
            break
        move = black.get_move(board)
        engine.move(board, move)
        if board.is_game_over():
            break
    win = {
        '1-0': 1,
        '1/2-1/2': 0,
        '0-1': -1,
    }[board.result()]
    data = []
    for i, (board, move) in enumerate(zip(boards, moves)):
        rotate = i % 2 == 1
        data.append((board, move, win, i, len(moves), rotate))
    return data


INT2CHR = '.PRNBQKprnbqk'


CHR2INT = {}
for i, c in enumerate(INT2CHR):
    CHR2INT[c] = i


def parse_board(text):
    text = ''.join(text.split())
    assert len(text) == 64
    nn = list(map(lambda c: CHR2INT[c], text))
    return np.array(nn, dtype='int8').reshape((8, 8))


SCALE_SURENESS = 1. / np.log(0 + 2)


def get_sureness(moves_before_result):
    return 1. / np.log((moves_before_result / 10.) ** 3. + 2.) / SCALE_SURENESS


def shuffle(x, y):
    data = list(zip(x, y))
    np.random.shuffle(data)
    x, y = zip(*data)
    x = np.array(x, dtype=x[0].dtype)
    y = np.array(y, dtype=y[0].dtype)
    return x, y


def transform(data):
    piece_in = []
    piece_out = []
    target_in = []
    target_out = []
    win_in = []
    win_out = []
    for board, move, win, move_index, num_moves, rotate in data:
        if rotate:
            board = board.swapcase()
        board = parse_board(board)
        from_yx = move.from_square // 8, move.from_square % 8
        to_yx = move.to_square // 8, move.to_square % 8
        if rotate:
            board = np.rot90(np.rot90(board))
            from_yx = 7 - from_yx[0], 7 - from_yx[1]
            to_yx = 7 - to_yx[0], 7 - to_yx[1]
            win *= -1

        board = np.equal.outer(np.arange(13), board).astype('float32')
        piece = np.zeros((64,), dtype='float32')
        piece_index = from_yx[0] * 8 + from_yx[1]
        piece[piece_index] = 1.
        piece_in.append(board)
        piece_out.append(piece)

        piece = np.zeros((1, 8, 8), dtype='float32')
        piece[0, from_yx[0], from_yx[1]] = 1.
        board_cat_piece = np.concatenate([board, piece], 0)
        target = np.zeros((64,), dtype='float32')
        target_index = to_yx[0] * 8 + to_yx[1]
        target[target_index] = 1.
        target_in.append(board_cat_piece)
        target_out.append(target)

        moves_before_result = num_moves - move_index
        sureness = get_sureness(moves_before_result)
        assert 0. < sureness <= 1.
        goodness = sureness * win
        goodness = np.array([goodness], dtype='float32')
        win_in.append(board)
        win_out.append(goodness)

    piece_in, piece_out = shuffle(piece_in, piece_out)
    target_in, target_out = shuffle(target_in, target_out)
    win_in, win_out = shuffle(win_in, win_out)
    return (piece_in, piece_out), (target_in, target_out), (win_in, win_out)


def train_from_play(process, time_per_move, piece_model, target_model,
                    win_model, num_batches=64, batch_size=16, val_frac=0.25):
    data = []
    z = num_batches * batch_size
    i = 0
    print('Observing games...')
    while len(data) < z:
        data += play_a_game(process, time_per_move)
        i += 1
        print('  %3d %5d' % (i, len(data)))
    data = data[:z]
    split = int(len(data) * (1. - val_frac))
    train = transform(data[:split])
    val = transform(data[split:])
    print('Piece predictor')
    piece_model.train_classifier((train[0], val[0]), stop=1, verbose=2)
    print('Target predictor')
    target_model.train_classifier((train[1], val[1]), stop=1, verbose=2)
    print('Win predictor')
    win_model.train_regressor((train[2], val[2]), stop=1, verbose=2)
    print()


conv = lambda n: Conv(n) > BatchNorm > ReLU > Z
cnn = lambda n: conv(n) > conv(n) > conv(n) > Z


def get_piece_model():
    return Input((13, 8, 8)) > cnn(256) > Conv(1) > Flatten > Softmax > Z


def get_target_model():
    return Input((14, 8, 8)) > cnn(256) > Conv(1) > Flatten > Softmax > Z


def get_win_model():
    return Input((13, 8, 8)) > cnn(256) > conv(1) > Flatten > Dense(1) > \
        Tanh > Z


def run(args):
    process = chess.uci.popen_engine(args.engine_binary)
    piece_model = get_piece_model()
    target_model = get_target_model()
    win_model = get_win_model()
    while True:
        train_from_play(process, args.time_per_move, piece_model, target_model,
                        win_model)


if __name__ == '__main__':
    run(parse_args())
