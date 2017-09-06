import numpy as np
from time import time

from .transform import Transform


def one_hot(x, vocab_size, dtype='float32'):
    shape = len(x), vocab_size
    r = np.zeros(shape).astype(dtype)
    for i, n in enumerate(x):
        r[i, n] = 1
    return r


class OneHot(Transform):
    def __init__(self, vocab_size=None, dtype='float32'):
        self.vocab_size = vocab_size
        self.dtype = dtype

    def fit(self, x, verbose=0, depth=0):
        if self.vocab_size is None:
            self.vocab_size = int(x.max() + 1)
        else:
            assert x.max() + 1 <= self.vocab_size

    def transform(self, x, verbose=0, depth=0):
        t0 = time()
        ret = one_hot(x, self.vocab_size, self.dtype)
        t = time() - t0
        self.done(t, verbose, depth)
        return ret

    def inverse_transform(self, x):
        return x.argmax(axis=1)
