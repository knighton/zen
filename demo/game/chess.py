from argparse import ArgumentParser
import numpy as np

from zen.dataset.kingbase_chess import Board, load_chess_piece_selection, \
    load_chess_target_selection, _yx_from_a1, a1_from_yx
from zen.layer import *  # noqa
from zen.model import Graph


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--load_verbose', type=int, default=2)
    ap.add_argument('--val_frac', type=int, default=0.2)
    ap.add_argument('--train_verbose', type=int, default=2)
    ap.add_argument('--train_piece_opt', type=str, default='adam')
    ap.add_argument('--train_piece_stop', type=int, default=2)
    ap.add_argument('--train_target_opt', type=str, default='adam')
    ap.add_argument('--train_target_stop', type=int, default=2)
    ap.add_argument('--play', type=str, default='console')
    return ap.parse_args()


conv = lambda n: Conv(n, 5, 2) > BatchNorm > ReLU > Z
cnn = lambda n: conv(n) > conv(n) > conv(n) > conv(n) > conv(n) > conv(n) > Z


def get_piece_model():
    return Input((13, 8, 8)) > cnn(64) > Conv(1) > Flatten > Softmax > Z


def get_target_model():
    return Input((14, 8, 8)) > cnn(64) > Conv(1) > Flatten > Softmax > Z


def show_board(board, heatmap, selected_yx):
    text = board.to_color_text(heatmap, selected_yx)
    lines = text.strip().split('\n')
    lines = list(map(lambda line: ' ' * 4 + line + '\n', lines))
    print()
    print(''.join(lines))


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
    return a1_from_yx(yx)


def select_piece(piece_model, board, strategy):
    indexes = board.to_numpy()
    board_arr = np.equal.outer(np.arange(13), indexes).astype('float32')
    board_arr = board_arr.reshape((1, 13, 8, 8))
    heatmap, = piece_model.predict_on_batch([board_arr])
    piece = select(heatmap, strategy)
    return board_arr, heatmap.reshape((8, 8)), piece


def select_target(target_model, board, board_arr, piece, strategy):
    piece_yx = _yx_from_a1(piece)
    piece_arr = np.zeros((1, 1, 8, 8), dtype='float32')
    piece_arr[0, 0, piece_yx[0], piece_yx[1]] = 1.
    board_from_arr = np.concatenate([board_arr, piece_arr], 1)
    heatmap, = target_model.predict_on_batch([board_from_arr])
    target = select(heatmap, strategy)
    return heatmap.reshape((8, 8)), target


def move(board, piece, target):
    piece = _yx_from_a1(piece)
    target = _yx_from_a1(target)
    board.move(piece, target, True, False, None)


def input_with_default(text, default):
    s = input(text)
    if not s:
        s = default
    return s


def console(piece_model, target_model, my_strategy='best',
            their_strategy='sample'):
    board = Board.initial()
    while True:
        board_arr, piece_heatmap, default_piece = \
            select_piece(piece_model, board, my_strategy)
        show_board(board, piece_heatmap, None)
        piece = input_with_default('move piece at (%s): ' % default_piece,
                                   default_piece)

        target_heatmap, default_target = select_target(
            target_model, board, board_arr, piece, my_strategy)
        show_board(board, target_heatmap, piece)
        target = input_with_default('move piece to (%s): ' % default_target,
                                    default_target)

        move(board, piece, target)

        board.rotate()

        board_arr, _, piece = select_piece(piece_model, board, their_strategy)
        _, target = select_target(target_model, board, board_arr, piece,
                                  their_strategy)
        print('\nThe computer moved from %s to %s.' % (piece, target))
        move(board, piece, target)

        board.rotate()


def run(args):
    piece_model = get_piece_model()
    target_model = get_target_model()

    print('Training piece selection model...')
    piece_data = load_chess_piece_selection(args.val_frac, args.load_verbose)
    piece_model.train_classifier(
        piece_data, opt=args.train_piece_opt, stop=args.train_piece_stop,
        verbose=args.train_verbose)

    print('Training target selection model...')
    target_data = load_chess_target_selection(args.val_frac, args.load_verbose)
    target_model.train_classifier(
        target_data, opt=args.train_target_opt, stop=args.train_target_stop,
        verbose=args.train_verbose)

    if args.play == 'console':
        console(piece_model, target_model)


if __name__ == '__main__':
    run(parse_args())
