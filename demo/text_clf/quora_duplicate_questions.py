from argparse import ArgumentParser
import random
import string
import sys

from zen.dataset.quora_duplicate_questions import load_quora_duplicate_questions
from zen.layer import *
from zen.model import Graph
from zen.transform import *


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--load_verbose', type=int, default=2)
    ap.add_argument('--shuffle', type=int, default=1)
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--train_verbose', type=int, default=2)
    ap.add_argument('--model', type=str, default='cnn')
    ap.add_argument('--opt', type=str, default='adam')
    ap.add_argument('--stop', type=int, default=1000)
    return ap.parse_args()


def cnn(question_len, vocab_size):
    conv = lambda n: Conv(n) > BatchNorm > ReLU > MaxPool > Z
    cnn = conv(16) > conv(16) > conv(16) > conv(16) > Z
    q1 = Input((question_len,), dtype='int64')
    q2 = Input((question_len,), dtype='int64')
    label = Concat()(q1, q2) > Embed(vocab_size, 16) > cnn > Flatten > \
            Dense(1) > Sigmoid > Z
    return Graph([q1, q2], label)


def transform(data, shuffle, val_frac, question_pipe, label_pipe, verbose):
    if shuffle:
        samples = list(zip(*data))
        random.shuffle(samples)
        data = list(zip(*samples))

    q1, q2, y = data
    split = int(len(q1) * val_frac)

    q1_train = q1[split:]
    q2_train = q2[split:]
    y_train = y[split:]
    q1_q2_train = question_pipe.fit_transform(q1_train + q2_train, verbose)
    n = len(q1_q2_train) // 2
    q1_train = q1_q2_train[:n]
    q2_train = q1_q2_train[n:]
    y_train = label_pipe.fit_transform(y_train, verbose)
    train = (q1_train, q2_train), y_train

    q1_val = question_pipe.transform(q1[:split], verbose)
    q2_val = question_pipe.transform(q2[:split], verbose)
    y_val = label_pipe.transform(y[:split], verbose)
    val = (q1_val, q2_val), y_val

    return train, val


def run(args):
    module = sys.modules[__name__]
    build = getattr(module, args.model)
    question_pipe = Pipe(Lower, Filter(string.ascii_lowercase + ' '), Split,
                         Length(16), Dict, NDArray('int64'))
    label_pipe = Pipe(NDArray('float32'))
    data = load_quora_duplicate_questions()
    data = transform(data, args.shuffle, args.val_frac, question_pipe,
                     label_pipe, args.load_verbose)
    question_len = data[0][0][0].shape[1]
    vocab_size = int(max(data[0][0][0].max(), data[0][0][1].max())) + 1
    if args.train_verbose:
        print('Train: question 1 %s' % (data[0][0][0].shape,))
        print('       question 2 %s' % (data[0][0][1].shape,))
        print('       label      %s' % (data[0][1].shape,))
        print('Val: question 1 %s' % (data[1][0][0].shape,))
        print('     question 2 %s' % (data[1][0][1].shape,))
        print('     label      %s' % (data[1][1].shape,))
        print('Vocab size: %d' % vocab_size)
    model = build(question_len, vocab_size)
    model.train_classifier(data, opt=args.opt, stop=args.stop,
                           verbose=args.train_verbose)


if __name__ == '__main__':
    run(parse_args())
