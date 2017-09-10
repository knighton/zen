from argparse import ArgumentParser
import string
import sys

from zen.dataset.nlvr import load_nlvr
from zen.layer import *
from zen.model import Graph
from zen.transform import *


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--load_verbose', type=int, default=0)
    ap.add_argument('--train_verbose', type=int, default=2)
    ap.add_argument('--model', type=str, default='cnn_cnn_mlp')
    ap.add_argument('--opt', type=str, default='adam')
    ap.add_argument('--stop', type=int, default=1000)
    return ap.parse_args()


conv = lambda n: Conv(n) > BatchNorm > MaxPool > ReLU > Z
dense = lambda n: Dense(n) > BatchNorm > ReLU > Dropout(0.5) > Z


def cnn_cnn_mlp(image_shape, text_len, vocab_size):
    image = Input(image_shape)
    image_embedding = image > conv(16) > conv(16) > conv(16) > conv(16) > \
                      conv(16) > Flatten > Z

    text = Input((text_len,), dtype='int64')
    text_embedding = text > Embed(vocab_size, 64) > conv(32) > conv(32) > \
                     Flatten > Z

    label = Concat()(image_embedding, text_embedding) > dense(256) > \
            dense(64) > dense(1) > Sigmoid > Z
    return Graph([image, text], label)


def transform_images(x):
    x = x.astype('float32')
    x -= x.mean()
    x /= x.std()
    return x


def show(data, split):
    print('%s: images %s' % (split, data[0][0].shape))
    print('%s  text   %s' % (' ' * len(split), data[0][1].shape))
    print('%s  labels %s' % (' ' * len(split), data[1].shape))


def transform(data, text_pipe, label_pipe, verbose):
    train, val = data
    train_images, train_texts, train_labels = train
    train_images = transform_images(train_images)
    train_texts = text_pipe.fit_transform(train_texts, verbose)
    train_labels = label_pipe.fit_transform(train_labels, verbose)
    val_images, val_texts, val_labels = val
    val_images = transform_images(val_images)
    val_texts = text_pipe.transform(val_texts, verbose)
    val_labels = label_pipe.transform(val_labels, verbose)
    train = (train_images, train_texts), train_labels
    val = (val_images, val_texts), val_labels
    return train, val


def run(args):
    module = sys.modules[__name__]
    build = getattr(module, args.model)
    text_pipe = Pipe(Lower(), Filter(string.ascii_lowercase + ' '), Split(),
                     Length(16), Dict(), NDArray('int64'))
    label_pipe = Pipe(TrueFalse(), NDArray('float32'))
    data = load_nlvr(args.load_verbose)
    data = transform(data, text_pipe, label_pipe, args.load_verbose)
    if args.train_verbose:
        show(data[0], 'Train')
        show(data[1], 'Val')
    images, sentences = data[0][0]
    image_shape = images.shape[1:]
    sentence_len = sentences.shape[1]
    vocab_size = int(sentences.max()) + 1
    model = build(image_shape, sentence_len, vocab_size)
    if hasattr(model, 'summary'):
        model.summary()
    model.train_classifier(data, stop=args.stop, verbose=args.train_verbose)


if __name__ == '__main__':
    run(parse_args())
