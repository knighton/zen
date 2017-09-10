from argparse import ArgumentParser
import sys

from zen.dataset.us_names import load_us_name_gender
from zen.layer import *
from zen.transform import *


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--max_name_len', type=int, default=16)
    ap.add_argument('--samples_per_epoch', type=int, default=64 * 1024)
    ap.add_argument('--verbose', type=int, default=2)
    ap.add_argument('--model', type=str, default='cnn')
    ap.add_argument('--opt', type=str, default='adam')
    ap.add_argument('--stop', type=int, default=1000)
    return ap.parse_args()


def cnn(review_len, vocab_size):
    conv = lambda n: Conv(n) > BatchNorm > ReLU > MaxPool > Z
    return Input((review_len,), dtype='int64') > Embed(vocab_size, 16) > \
           conv(16) > conv(16) > conv(16) > conv(16) > Flatten > Dense(1) > \
           Sigmoid > Z


def run(args):
    module = sys.modules[__name__]
    build = getattr(module, args.model)
    data = load_us_name_gender(args.max_name_len, args.samples_per_epoch,
                               args.verbose)
    text_len = data.get_sample_shapes()[0][0][0]
    model = build(text_len, 128)
    model.train_classifier(data, opt=args.opt, stop=args.stop,
                           verbose=args.verbose)


if __name__ == '__main__':
    run(parse_args())
