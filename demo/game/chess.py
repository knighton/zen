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


def console(piece_model, target_model):
    board = Board.initial()
    while True:
        indexes = board.to_numpy()
        board_arr = np.equal.outer(np.arange(13), indexes).astype('float32')
        board_arr = board_arr.reshape((1, 13, 8, 8))
        heatmap, = piece_model.predict_on_batch([board_arr])
        heatmap = heatmap.reshape((8, 8))
        show_board(board, heatmap, None)

        n = heatmap.argmax()
        best_from_yx = n // 8, n % 8
        best_from_a1 = a1_from_yx(best_from_yx)
        from_a1 = input('move piece at (%s): ' % best_from_a1)
        if not from_a1:
            from_a1 = best_from_a1
        from_yx = _yx_from_a1(from_a1)
        from_arr = np.zeros((1, 1, 8, 8), dtype='float32')
        from_arr[0, 0, from_yx[0], from_yx[1]] = 1.
        board_from_arr = np.concatenate([board_arr, from_arr], 1)
        heatmap, = target_model.predict_on_batch([board_from_arr])
        heatmap = heatmap.reshape((8, 8))
        show_board(board, heatmap, from_yx)

        n = heatmap.argmax()
        best_to_yx = n // 8, n % 8
        best_to_a1 = a1_from_yx(best_to_yx)
        to_a1 = input('move piece to (%s): ' % best_to_a1)
        if not to_a1:
            to_a1 = best_to_a1
        to_yx = _yx_from_a1(to_a1)
        board.move(from_yx, to_yx, True, False, None)

        board.rotate()

        indexes = board.to_numpy()
        board_arr = np.equal.outer(np.arange(13), indexes).astype('float32')
        board_arr = board_arr.reshape((1, 13, 8, 8))
        heatmap, = piece_model.predict_on_batch([board_arr])
        n = heatmap.argmax()
        best_from_yx = n // 8, n % 8

        from_arr = np.zeros((1, 1, 8, 8), dtype='float32')
        from_arr[0, 0, best_from_yx[0], best_from_yx[1]] = 1.
        board_from_arr = np.concatenate([board_arr, from_arr], 1)
        heatmap, = target_model.predict_on_batch([board_from_arr])
        n = heatmap.argmax()
        best_to_yx = n // 8, n % 8

        board.move(best_from_yx, best_to_yx, True, False, None)

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
