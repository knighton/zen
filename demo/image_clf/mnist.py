from argparse import ArgumentParser

from zen.dataset.mnist import load_mnist
from zen.layer import *
from zen.transform.one_hot import one_hot


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--verbose', type=int, default=2)
    ap.add_argument('--stop', type=int, default=1000)
    return ap.parse_args()


def run(args):
    (x_train, y_train), (x_val, y_val) = load_mnist(args.verbose)
    image_shape = x_train.shape[1:]
    num_classes = int(y_train.max() + 1)
    y_train = one_hot(y_train, num_classes)
    y_val = one_hot(y_val, num_classes)
    data = (x_train, y_train), (x_val, y_val)
    model = Sequence(
        Input(image_shape),
        Flatten,
        Dense(256),
        ReLU,
        Dense(64),
        ReLU,
        Dense(num_classes),
        Softmax
    )
    model.train_classifier(data, stop=args.stop)


if __name__ == '__main__':
    run(parse_args())
