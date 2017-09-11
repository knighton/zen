from argparse import ArgumentParser

from zen.dataset.higgs_boson import load_higgs_boson
from zen.layer import *


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--scale', type=int, default=1)
    ap.add_argument('--verbose', type=int, default=2)
    ap.add_argument('--opt', type=str, default='adam')
    ap.add_argument('--stop', type=int, default=1000)
    return ap.parse_args()


def mlp(num_features):
    layer = lambda n: Dense(n) > BatchNorm > ReLU > Dropout(0.5) > Z
    mlp = layer(64) > layer(64) > layer(64) > layer(8) > Z
    in_shape = num_features,
    return Input(in_shape) > mlp > Dense(1) > Sigmoid > Z


def run(args):
    data = load_higgs_boson(args.verbose)
    num_features = data[0][0].shape[1]
    model = mlp(num_features)
    model.train_classifier(data, opt=args.opt, stop=args.stop)


if __name__ == '__main__':
    run(parse_args())
