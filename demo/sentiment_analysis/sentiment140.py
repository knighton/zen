from argparse import ArgumentParser
import string
import sys

from zen.dataset.sentiment140 import load_sentiment140
from zen.layer import *
from zen.transform import *


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--verbose', type=int, default=2)
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--model', type=str, default='cnn')
    ap.add_argument('--opt', type=str, default='adam')
    ap.add_argument('--stop', type=int, default=1000)
    return ap.parse_args()


def cnn(review_len, vocab_size):
    conv = lambda n: Conv(n) > BatchNorm > ReLU > MaxPool > Z
    cnn = []
    for n in [32] * 4:
        cnn.append(conv(n))
    cnn = Sequence(*cnn)
    in_shape = review_len,
    return Input(in_shape, dtype='int64') > Embed(vocab_size, 64) > cnn > \
        Flatten > Dense(1) > Sigmoid > Z


def eru(review_len, vocab_size):
    return Input((review_len,), dtype='int64') > Embed(vocab_size, 64) > \
        ERU(64) > Dropout(0.1) > ERU(64, ret='last') > Dropout(0.1) > \
        Dense(1) > Sigmoid > Z


def transform(data, text_pipe, label_pipe, verbose):
    train_texts = text_pipe.fit_transform(data[0][0], verbose)
    train_labels = label_pipe.fit_transform(data[0][1], verbose)
    val_texts = text_pipe.transform(data[1][0], verbose)
    val_labels = label_pipe.transform(data[1][1], verbose)
    return (train_texts, train_labels), (val_texts, val_labels)


def run(args):
    module = sys.modules[__name__]
    build = getattr(module, args.model)
    data = load_sentiment140(args.val_frac, args.verbose)
    text_pipe = Pipe(Lower, Filter(string.ascii_lowercase + ' '), Split,
                     Length(32), Dict, NDArray('int64'))
    label_pipe = Pipe(NDArray('float32'))
    data = transform(data, text_pipe, label_pipe, args.verbose)
    review_len = data[0][0].shape[1]
    vocab_size = int(data[0][0].max() + 1)
    model = build(review_len, vocab_size)
    model.train_classifier(data, opt=args.opt, stop=args.stop)


if __name__ == '__main__':
    run(parse_args())
