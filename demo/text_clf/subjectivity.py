from argparse import ArgumentParser
import string
import sys

from zen.dataset.subjectivity import load_subjectivity
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
    conv = lambda n: SequenceSpec(Conv(n), BatchNorm, ReLU, MaxPool)
    cnn = []
    for n in [32] * 4:
        cnn.append(conv(n))
    cnn = SequenceSpec(*cnn)
    in_shape = review_len,
    spec = SequenceSpec(
        Input(in_shape, dtype='int64'), Embed(vocab_size, 64), cnn, Flatten,
        Dense(1), Sigmoid)
    model, out_shape, out_dtype = spec.build()
    return model


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
