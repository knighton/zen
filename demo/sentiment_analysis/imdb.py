from argparse import ArgumentParser
import string
import sys

from zen.dataset.imdb import load_imdb
from zen.layer import *
from zen.transform import *


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--dataset', type=int, default=10)
    ap.add_argument('--cifar10_val_frac', type=float, default=0.2)
    ap.add_argument('--verbose', type=int, default=2)
    ap.add_argument('--model', type=str, default='cnn')
    ap.add_argument('--stop', type=int, default=1000)
    return ap.parse_args()


def cnn(review_len, vocab_size):
    conv = lambda n: SequenceSpec(Conv(n), BatchNorm, ReLU, MaxPool)
    cnn = []
    for n in [32] * 8:
        cnn.append(conv(n))
    cnn = SequenceSpec(*cnn)
    input_shape = review_len,
    spec = SequenceSpec(
        Input(input_shape, dtype='int64'), Embed(vocab_size, 64), cnn, Flatten,
        Dense(1), Sigmoid)
    model, out_shape, out_dtype = spec.build()
    return model


def transform(data, text_pipe, label_pipe):
    train_texts = text_pipe.fit_transform(data[0][0])
    train_labels = label_pipe.fit_transform(data[0][1])
    val_texts = text_pipe.transform(data[1][0])
    val_labels = label_pipe.transform(data[1][1])
    data = (train_texts, train_labels), (val_texts, val_labels)
    return data


def run(args):
    module = sys.modules[__name__]
    build = getattr(module, args.model)
    data = load_imdb(args.verbose)
    text_pipe = Pipe(Lower, Filter(string.ascii_lowercase + ' '), Split,
                     Length(512), Dict, NDArray('int64'))
    label_pipe = Pipe(NDArray('float32'))
    data = transform(data, text_pipe, label_pipe)
    review_len = data[0][0].shape[1]
    vocab_size = int(data[0][0].max() + 1)
    model = build(review_len, vocab_size)
    model.train_classifier(data, stop=args.stop)


if __name__ == '__main__':
    run(parse_args())
