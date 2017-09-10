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
    ap.add_argument('--stop', type=int, default=20)
    return ap.parse_args()


def cnn(review_len, vocab_size):
    conv = lambda n: Conv(n) > BatchNorm > ReLU > Z
    block = lambda n: conv(n) > conv(n) > MaxPool > Z
    return Input((review_len,), dtype='int64') > Embed(vocab_size, 64) > \
           block(64) > block(64) > block(64) > block(64) > Flatten > \
           Dense(1) > Sigmoid > Z


def name_to_ints(s, text_len):
    nn = list(map(ord, s))
    nn = list(map(lambda n: n if n < 128 else 1, nn))
    nn = nn[:text_len]
    nn += [0] * (text_len - len(nn))
    return np.array(nn, dtype='int64')


def to_chr(n):
    if not n:
        c = ' '
    elif n == 1:
        c = '.'
    else:
        c = chr(n)
    return c


def ints_to_name(nn):
    return ''.join(map(to_chr, nn)).strip()


def show_name(xx, yy_pred, yy_true):
    for x, y_pred, y_true in zip(xx[0], yy_pred[0], yy_true[0]):
        pct_male = y_pred[0] * 100.
        mf = {
            0: 'F',
            1: 'M',
        }[y_true[0]]
        name = ints_to_name(x)
        is_correct = y_pred[0].round() == y_true[0]
        is_correct = ' ' if is_correct else 'x'
        print('* %5.2f%% %s %s %s' % (pct_male, mf, is_correct, name))


def show_examples(model, data):
    i = 0
    for xx, yy_true, is_training in data.each_batch(32):
        if i == 5:
            break
        yy_pred = model.predict_on_batch(xx)
        show_name(xx, yy_pred, yy_true)
        i += 1


def demo(model, max_name_len):
    while True:
        name = input('Name: ')
        ints = name_to_ints(name, max_name_len)
        x = np.expand_dims(ints, 0)
        xx = [x]
        yy_pred = model.predict_on_batch(xx)
        y = yy_pred[0]
        if y < 0.5:
            y = 1. - y
            gender = 'female'
        else:
            gender = 'male'
        print('%.2f%% %s' % (y * 100., gender))


def run(args):
    module = sys.modules[__name__]
    build = getattr(module, args.model)
    data = load_us_name_gender(args.max_name_len, args.samples_per_epoch,
                               args.verbose)
    text_len = data.get_sample_shapes()[0][0][0]
    model = build(text_len, 128)
    model.train_classifier(data, opt=args.opt, stop=args.stop,
                           verbose=args.verbose)
    show_examples(model, data)
    demo(model, args.max_name_len)


if __name__ == '__main__':
    run(parse_args())
