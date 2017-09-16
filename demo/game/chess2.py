from argparse import ArgumentParser

from zen.dataset.chess import load_chess
from zen.layer import *  # noqa


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--keep_frac', type=float, default=0.1)
    ap.add_argument('--samples_per_epoch', type=int, default=100000)
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--load_verbose', type=int, default=2)
    ap.add_argument('--train_piece_stop', type=int, default=5)
    ap.add_argument('--train_target_stop', type=int, default=5)
    ap.add_argument('--train_board_stop', type=int, default=5)
    ap.add_argument('--train_verbose', type=int, default=2)
    return ap.parse_args()


conv = lambda n: Conv(n) > BatchNorm > ReLU > Z
cnn = lambda n: conv(n) > conv(n) > conv(n) > conv(n) > conv(n) > Z


def get_piece_model():
    return Input((13, 8, 8)) > cnn(256) > Conv(1) > Flatten > Softmax > Z


def get_target_model():
    return Input((14, 8, 8)) > cnn(256) > Conv(1) > Flatten > Softmax > Z


def get_win_model():
    return Input((13, 8, 8)) > cnn(256) > conv(1) > Flatten > Dense(1) > \
        Tanh > Z


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


if __name__ == '__main__':
    run(parse_args())
