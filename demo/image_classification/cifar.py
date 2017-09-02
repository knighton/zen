from argparse import ArgumentParser
from random import shuffle

from zen.dataset.cifar import load_cifar
from zen.layer import *
from zen.transform.one_hot import one_hot


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--dataset', type=int, default=10)
    ap.add_argument('--cifar10_val_frac', type=float, default=0.2)
    ap.add_argument('--verbose', type=int, default=2)
    ap.add_argument('--model', type=str, default='cnn')
    ap.add_argument('--stop', type=int, default=1000)
    return ap.parse_args()


def mlp(image_shape, num_classes):
    layer = lambda n: SequenceSpec(Dense(n), BatchNorm, ReLU, Dropout(0.5))
    mlp = SequenceSpec(layer(256), layer(64))
    spec = SequenceSpec(
        Input(image_shape), Flatten, mlp, Dense(num_classes), Softmax)
    model, out_shape, out_dtype = spec.build()
    return model


def cnn(image_shape, num_classes):
    layer = lambda n: Sequence(
        Conv(n), BatchNorm, ReLU, SpatialDropout(0.25), MaxPool)
    cnn = layer(16), layer(32), layer(64), layer(128)
    spec = SequenceSpec(
        Input(image_shape), cnn, Flatten, Dense(num_classes), Softmax)
    model, out_shape, out_dtype = spec.build()
    return model


def run(args):
    data = load_cifar(args.dataset, args.cifar10_val_frac, args.verbose)
    image_shape = data.train[0].shape[1:]
    num_classes = int(data.train[1].max() + 1)
    train = data.train[0], one_hot(data.train[1], num_classes)
    val = data.val[0], one_hot(data.val[1], num_classes)
    if args.verbose:
        ss = []
        for i, label in enumerate(data.labels):
            ss.append('%s (%d)' % (label, i))
        print('Classes: %s.' % ', '.join(ss))
    model = mlp(image_shape, num_classes)
    model.train_classifier((train, val), stop=args.stop)


if __name__ == '__main__':
    run(parse_args())
