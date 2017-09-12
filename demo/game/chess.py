from argparse import ArgumentParser

from zen.dataset.kingbase_chess import load_chess_moves_select_piece, \
    load_chess_moves_select_dest
from zen.layer import *  # noqa
from zen.model import Graph


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--load_verbose', type=int, default=2)
    ap.add_argument('--val_frac', type=int, default=0.2)
    ap.add_argument('--train_verbose', type=int, default=2)
    ap.add_argument('--train_piece_opt', type=str, default='adam')
    ap.add_argument('--train_piece_stop', type=int, default=10)
    ap.add_argument('--train_dest_opt', type=str, default='adam')
    ap.add_argument('--train_dest_stop', type=int, default=10)
    return ap.parse_args()


conv = lambda n: Conv(n, 5, 2) > BatchNorm > ReLU > Z
cnn = lambda n: conv(n) > conv(n) > conv(n) > conv(n) > conv(n) > conv(n) > Z


def get_piece_model():
    return Input((64,), dtype='int64') > Embed(13, 32) > \
        Reshape((-1, 8, 8)) > cnn(64) > Conv(1) > Flatten > Softmax > Z


def get_dest_model():
    board = Input((64,), dtype='int64')
    selected_piece = Input((64,))
    board_embeddings = board > Embed(13, 32) > Z
    selected_piece_reshaped = selected_piece > Reshape((1, -1)) > Z
    dest = Concat()(board_embeddings, selected_piece_reshaped) > \
        Reshape((-1, 8, 8)) > cnn(64) > Flatten > Dense(64) > Softmax > Z
    return Graph([board, selected_piece], dest)


def run(args):
    piece_model = get_piece_model()
    dest_model = get_dest_model()

    print('Training piece selection model...')
    piece_data = load_chess_moves_select_piece(args.val_frac, args.load_verbose)
    piece_model.train_classifier(
        piece_data, opt=args.train_piece_opt, stop=args.train_piece_stop,
        verbose=args.train_verbose)

    print('Training destination selection model...')
    dest_data = load_chess_moves_select_dest(args.val_frac, args.load_verbose)
    dest_model.train_classifier(
        dest_data, opt=args.train_dest_opt, stop=args.train_dest_stop,
        verbose=args.train_verbose)


if __name__ == '__main__':
    run(parse_args())
