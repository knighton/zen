from argparse import ArgumentParser
from random import shuffle

from zen.dataset.cifar import load_cifar10
from zen.layer import *
from zen.transform.one_hot import one_hot


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--verbose', type=int, default=2)
    ap.add_argument('--stop', type=int, default=1000)
    return ap.parse_args()


def split(x, y, val_frac):
    samples = list(zip(x, y))
    shuffle(samples)
    split = int(len(samples) * val_frac)
    x_train = x[split:]
    x_val = x[:split]
    y_train = y[split:]
    y_val = y[:split]
    return (x_train, y_train), (x_val, y_val)


def run(args):
    val_frac = 0.2
    data = load_cifar10(args.verbose)
    train, val = split(data.x, data.y, val_frac)
    image_shape = train[0].shape[1:]
    num_classes = train[1].max() + 1
    train = train[0], one_hot(train[1], num_classes)
    val = val[0], one_hot(val[1], num_classes)

    spec = SequenceSpec(
        Input(image_shape),
        Flatten,
        Dense(256),
        ReLU,
        Dense(64),
        ReLU,
        Dense(num_classes),
        Softmax
    )

    model, out_shape, out_dtype = spec.build()

    model.train_classifier((train, val), stop=args.stop)


if __name__ == '__main__':
    run(parse_args())
