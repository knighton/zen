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
    layer = lambda n: SequenceSpec(Dense(n), BatchNorm, ReLU, Dropout(0.5))
    mlp = SequenceSpec(layer(64), layer(64), layer(64), layer(8))
    in_shape = num_features,
    spec = SequenceSpec(Input(in_shape), mlp, Dense(1), Sigmoid)
    model, out_shape, out_dtype = spec.build()
    return model


def run(args):
    data = load_higgs_boson(args.verbose)
    num_features = data[0][0].shape[1]
    model = mlp(num_features)
    model.train_classifier(data, opt=args.opt, stop=args.stop)


if __name__ == '__main__':
    run(parse_args())
