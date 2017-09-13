from argparse import ArgumentParser

from zen.dataset.kingbase_chess import load_chess_moves_select_piece, \
    load_chess_moves_select_target
from zen.layer import *  # noqa
from zen.model import Graph


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--load_verbose', type=int, default=2)
    ap.add_argument('--val_frac', type=int, default=0.2)
    ap.add_argument('--train_verbose', type=int, default=2)
    ap.add_argument('--train_piece_opt', type=str, default='adam')
    ap.add_argument('--train_piece_stop', type=int, default=1)
    ap.add_argument('--train_target_opt', type=str, default='adam')
    ap.add_argument('--train_target_stop', type=int, default=1)
    return ap.parse_args()


conv = lambda n: Conv(n, 5, 2) > BatchNorm > ReLU > Z
cnn = lambda n: conv(n) > conv(n) > conv(n) > conv(n) > conv(n) > conv(n) > Z


def get_piece_model():
    return Input((13, 8, 8)) > cnn(64) > Conv(1) > Flatten > Softmax > Z


def get_target_model():
    return Input((14, 8, 8)) > cnn(64) > Conv(1) > Flatten > Softmax > Z


def run(args):
    piece_model = get_piece_model()
    target_model = get_target_model()

    print('Training piece selection model...')
    piece_data = load_chess_moves_select_piece(args.val_frac, args.load_verbose)
    piece_model.train_classifier(
        piece_data, opt=args.train_piece_opt, stop=args.train_piece_stop,
        verbose=args.train_verbose)

    print('Training target selection model...')
    target_data = load_chess_moves_select_target(
        args.val_frac, args.load_verbose)
    target_model.train_classifier(
        target_data, opt=args.train_target_opt, stop=args.train_target_stop,
        verbose=args.train_verbose)


if __name__ == '__main__':
    run(parse_args())
