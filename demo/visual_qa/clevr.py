from argparse import ArgumentParser
import string
import sys

from zen.dataset.clevr import load_clevr
from zen.layer import *  # noqa
from zen.model import Graph
from zen.transform import *  # noqa


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--dataset', type=str, default='main')
    ap.add_argument('--storage', type=str, default='ram')
    ap.add_argument('--size', type=str, default='60x40')
    ap.add_argument('--load_verbose', type=int, default=2)
    ap.add_argument('--model', type=str, default='cnn_cnn_mlp')
    ap.add_argument('--stop', type=int, default=1000)
    ap.add_argument('--train_verbose', type=int, default=2)
    return ap.parse_args()


conv = lambda n: Conv(n) > BatchNorm > MaxPool > ReLU > Z
dense = lambda n: Dense(n) > BatchNorm > ReLU > Dropout(0.5) > Z


def cnn_cnn_mlp(image_shape, text_len, vocab_size, num_classes):
    image = Input(image_shape)
    image_embedding = image > conv(128) > conv(128) > conv(128) > conv(128) > \
        Shape('image output') > Flatten > Z

    text = Input((text_len,), dtype='int64')
    text_embedding = text > Embed(vocab_size, 128) > conv(128) > conv(128) > \
        conv(128) > Shape('text output') > Flatten > Z

    label = Concat()(image_embedding, text_embedding) > Shape('concat') > \
        Dropout(0.5) > dense(512) > dense(256) > dense(128) > \
        Dense(num_classes) > Softmax > Z
    return Graph([image, text], label)


def parse_size(text):
    size = tuple(map(int, text.split('x')))
    assert len(size) == 2
    assert 0 < size[0]
    assert 0 < size[1]
    return size


def run(args):
    module = sys.modules[__name__]
    build = getattr(module, args.model)
    question_pipe = Pipe(Lower, Filter(string.ascii_lowercase + ' '), Split,
                         Length(32), Dict, NDArray('int64'))
    answer_pipe = Pipe(Split, Dict, OneHot)
    size = parse_size(args.size)
    data = load_clevr(args.dataset, question_pipe, answer_pipe, args.storage,
                      size, args.load_verbose)
    image_shape = (3,) + size
    text_len = data.train.get_sample_shapes()[0][1][0]
    vocab_size = data.train.get_question_vocab_size()
    num_classes = data.train.get_answer_vocab_size()
    model = build(image_shape, text_len, vocab_size, num_classes)
    model.train_classifier(data, stop=args.stop, verbose=args.train_verbose)


if __name__ == '__main__':
    run(parse_args())
