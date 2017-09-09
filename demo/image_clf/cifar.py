from argparse import ArgumentParser
from random import shuffle
import sys

from zen.dataset.cifar import load_cifar
from zen.layer import *
from zen.model import Graph
from zen.transform.one_hot import one_hot


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--dataset', type=int, default=10)
    ap.add_argument('--cifar10_val_frac', type=float, default=0.2)
    ap.add_argument('--verbose', type=int, default=2)
    ap.add_argument('--model', type=str, default='cnn')
    ap.add_argument('--opt', type=str, default='adam')
    ap.add_argument('--stop', type=int, default=1000)
    return ap.parse_args()


def mlp_vee(image_shape, num_classes):
    layer = lambda n: Dense(n) > BatchNorm > ReLU > Dropout(0.5) > Z
    mlp = layer(256) > layer(64) > Z
    return Input(image_shape) > Flatten > mlp > Dense(num_classes) > Softmax > Z


def mlp_sequence(image_shape, num_classes):
    layer = lambda n: Sequence(Dense(n), BatchNorm, ReLU, Dropout(0.5))
    mlp = Sequence(layer(256), layer(64))
    return Sequence(
        Input(image_shape), Flatten, mlp, Dense(num_classes), Softmax)


def mlp_graph(image_shape, num_classes):
    def layer(x, n):
        x = Dense(n)(x)
        x = BatchNorm()(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        return x

    image = Input(image_shape)
    x = Flatten()(image)
    x = layer(x, 256)
    x = layer(x, 64)
    x = Dense(num_classes)(x)
    x = Softmax()(x)
    return Graph(image, x)


def cnn(image_shape, num_classes):
    layer = lambda n: Sequence(
        Conv(n), BatchNorm, ReLU, SpatialDropout(0.25), MaxPool)
    cnn = Sequence(layer(16), layer(32), layer(64), layer(128))
    return Sequence(
        Input(image_shape), cnn, Flatten, Dense(num_classes), Softmax)


def cnn_big(image_shape, num_classes):
    layer = lambda n: Sequence(
        Conv(n), BatchNorm, ReLU, SpatialDropout(0.25))
    block = lambda n: Sequence(layer(n), layer(n), MaxPool)
    cnn = Sequence(block(16), block(32), block(64), block(128))
    return Sequence(
        Input(image_shape), cnn, Flatten, Dense(num_classes), Softmax)


def run(args):
    module = sys.modules[__name__]
    build = getattr(module, args.model)
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
    model = build(image_shape, num_classes)
    model.train_classifier((train, val), opt=args.opt, stop=args.stop)


if __name__ == '__main__':
    run(parse_args())
