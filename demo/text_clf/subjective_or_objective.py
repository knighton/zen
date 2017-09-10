from argparse import ArgumentParser
import string
import sys

from zen.dataset.subjectivity import load_subjectivity
from zen.model import Graph
from zen.layer import *
from zen.transform import *


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--verbose', type=int, default=2)
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


def rnn(review_len, vocab_size):
    rnn = lambda n: BiLSTM(n, ret='last') > Dropout(0.5) > Z
    in_shape = review_len,
    return Input(in_shape, dtype='int64') > Embed(vocab_size, 32) > rnn(64) > \
           Flatten > Dense(1) > Sigmoid > Z


def cnn_rnn(review_len, vocab_size):
    in_ = Input((review_len,), dtype='int64')
    x = in_ > Embed(vocab_size, 64) > Z

    conv = lambda n: Conv(n) > BatchNorm > ReLU > MaxPool > Z
    cnn = x > conv(32) > conv(32) > conv(32) > conv(32) > Flatten > Z

    rnn = x > ERU(64) > Dropout(0.5) > ERU(64, ret='last') > Dropout(0.5) > Z

    dense = lambda n: Dense(n) > BatchNorm > ReLU > Dropout(0.75) > Z
    out = Concat(1)(cnn, rnn) > dense(256) > dense(32) > Dense(1) > Sigmoid > Z
    return Graph(in_, out)


def transform(data, text_pipe, label_pipe):
    train_texts = text_pipe.fit_transform(data[0][0])
    train_labels = label_pipe.fit_transform(data[0][1])
    val_texts = text_pipe.transform(data[1][0])
    val_labels = label_pipe.transform(data[1][1])
    return (train_texts, train_labels), (val_texts, val_labels)


def run(args):
    module = sys.modules[__name__]
    build = getattr(module, args.model)
    text_pipe = Pipe(Lower, Filter(string.ascii_lowercase + ' '), Split,
                     Length(32), Dict, NDArray('int64'))
    label_pipe = Pipe(NDArray('float32'))
    data = load_subjectivity(args.val_frac, args.verbose)
    data = transform(data, text_pipe, label_pipe)
    text_len = data[0][0].shape[1]
    vocab_size = int(data[0][0].max() + 1)
    model = build(text_len, vocab_size)
    model.train_classifier(data, opt=args.opt, stop=args.stop)


if __name__ == '__main__':
    run(parse_args())
