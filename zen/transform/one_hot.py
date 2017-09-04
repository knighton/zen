import numpy as np

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

    def fit(self, x):
        if self.vocab_size is None:
            self.vocab_size = int(x.max() + 1)
        else:
            assert x.max() + 1 <= self.vocab_size

    def transform(self, x):
        return one_hot(x, self.vocab_size, self.dtype)

    def inverse_transform(self, x):
        return x.argmax(axis=1)
