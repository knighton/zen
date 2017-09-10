from argparse import ArgumentParser
import string

from zen.dataset.clevr import load_clevr
from zen.layer import *
from zen.transform import *


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--dataset', type=str, default='main')
    ap.add_argument('--storage', type=str, default='ram')
    ap.add_argument('--size', type=str, default='60x40')
    ap.add_argument('--load_verbose', type=int, default=2)
    ap.add_argument('--train_verbose', type=int, default=2)
    return ap.parse_args()


def parse_size(text):
    size = tuple(map(int, text.split('x')))
    assert len(size) == 2
    assert 0 < size[0]
    assert 0 < size[1]
    return size


def run(args):
    question_pipe = Pipe(Lower, Filter(string.ascii_lowercase + ' '), Split,
                         Length(32), Dict, NDArray('int64'))
    answer_pipe = Pipe(Split, Dict, OneHot)
    size = parse_size(args.size)
    data = load_clevr(args.dataset, question_pipe, answer_pipe, args.storage,
                      size, args.load_verbose)


if __name__ == '__main__':
    run(parse_args())
