from argparse import ArgumentParser
import string
import sys

from zen.dataset.twenty_newsgroups import load_twenty_newsgroups
from zen.layer import *
from zen.transform import *


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--verbose', type=int, default=2)
    ap.add_argument('--model', type=str, default='cnn')
    ap.add_argument('--opt', type=str, default='adam')
    ap.add_argument('--stop', type=int, default=1000)
    return ap.parse_args()


def cnn(review_len, vocab_size, num_classes):
    conv = lambda n: Conv(n) > BatchNorm > ReLU > MaxPool > Z
    cnn = []
    for n in [32] * 8:
        cnn.append(conv(n))
    cnn = Sequence(*cnn)
    in_shape = review_len,
    return Input(in_shape, dtype='int64') > Embed(vocab_size, 32) > cnn > \
           Flatten > Dense(num_classes) > Softmax > Z


def eru(review_len, vocab_size, num_classes):
    in_shape = review_len,
    return Input(in_shape, dtype='int64') > Embed(vocab_size, 64) > ERU(64) > \
           Dropout(0.25) > ERU(64, ret='last') > Dropout(0.25) > \
           Dense(num_classes) > Softmax > Z


def gru(review_len, vocab_size, num_classes):
    in_shape = review_len,
    return Input(in_shape, dtype='int64') > Embed(vocab_size, 64) > \
           GRU(64, ret='last') > Dense(num_classes) > Softmax > Z


def lstm(review_len, vocab_size, num_classes):
    in_shape = review_len,
    return Input(in_shape, dtype='int64') > Embed(vocab_size, 64) > \
           LSTM(64, ret='last') > Dense(num_classes) > Softmax > Z


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
                     Length(512), Dict, NDArray('int64'))
    label_pipe = Pipe(OneHot)
    data = load_twenty_newsgroups(args.verbose)
    data = transform((data.train, data.test), text_pipe, label_pipe)
    review_len = data[0][0].shape[1]
    vocab_size = int(data[0][0].max() + 1)
    num_classes = data[0][1].shape[1]
    if args.verbose:
        print('Train texts:  %s' % (data[0][0].shape,))
        print('      labels: %s' % (data[0][1].shape,))
        print('Val texts:  %s' % (data[1][0].shape,))
        print('    labels: %s' % (data[1][1].shape,))
        print('Text vocab size: %d' % vocab_size)
    model = build(review_len, vocab_size, num_classes)
    model.train_classifier(data, opt=args.opt, stop=args.stop)


if __name__ == '__main__':
    run(parse_args())
