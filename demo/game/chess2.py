from argparse import ArgumentParser
import numpy as np

from zen.dataset.chess import Game, load_chess
from zen.dataset.chess.piece_type import PieceType
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
    yx = index // 8, index % 8
    return Game.yx_to_a1(yx)


def select_piece(piece_model, game, strategy):
    indexes = game.board
    board_arr = np.equal.outer(np.arange(13), indexes).astype('float32')
    board_arr = board_arr.reshape((1, 13, 8, 8))
    legal_mask = game.get_movable_pieces()
    heatmap, = piece_model.predict_on_batch([board_arr])
    heatmap = heatmap.reshape((8, 8))
    heatmap = heatmap * legal_mask
    heatmap /= heatmap.sum()
    piece_a1 = select(heatmap, strategy)
    return board_arr, heatmap, piece_a1


def select_target(target_model, game, board_arr, piece_a1, strategy):
    piece_yx = Game.a1_to_yx(piece_a1)
    piece_arr = np.zeros((1, 1, 8, 8), dtype='float32')
    piece_arr[0, 0, piece_yx[0], piece_yx[1]] = 1.
    board_from_arr = np.concatenate([board_arr, piece_arr], 1)
    legal_mask = game.get_possible_moves(piece_yx)
    heatmap, = target_model.predict_on_batch([board_from_arr])
    heatmap = heatmap.reshape((8, 8))
    heatmap = heatmap * legal_mask
    heatmap /= heatmap.sum()
    target_a1 = select(heatmap, strategy)
    return heatmap, target_a1


def input_with_default(text, default):
    s = input(text)
    if not s:
        s = default
    return s


def move(game, piece, target):
    piece = Game.a1_to_yx(piece)
    target = Game.a1_to_yx(target)
    if game.board[piece] == PieceType.my_pawn and target[0] == 7:
        promote_to = PieceType.my_queen
    else:
        promote_to = None
    game.apply_yx_yx_move(piece, target, promote_to)


def console(piece_model, target_model, win_model, my_strategy='best',
            their_strategy='sample'):
    game = Game.start()
    while True:
        board_arr, piece_heatmap, default_piece_a1 = \
            select_piece(piece_model, game, my_strategy)
        print(game.dump_board_pretty(piece_heatmap, None))
        piece_a1 = input_with_default('move piece at (%s): ' % default_piece_a1,
                                      default_piece_a1)

        target_heatmap, default_target_a1 = select_target(
            target_model, game, board_arr, piece_a1, my_strategy)
        print(game.dump_board_pretty(target_heatmap, piece_a1))
        target_a1 = input_with_default(
            'move piece to (%s): ' % default_target_a1, default_target_a1)

        move(game, piece_a1, target_a1)
        game.switch_sides()

        board_arr, _, piece_a1 = select_piece(piece_model, game, their_strategy)
        _, target_a1 = select_target(target_model, game, board_arr, piece_a1,
                                     their_strategy)
        move(game, piece_a1, target_a1)
        game.switch_sides()


def run(args):
    print('Constructing models...')
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
        console(piece_model, target_model, win_model)


if __name__ == '__main__':
    run(parse_args())
