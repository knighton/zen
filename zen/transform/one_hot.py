import numpy as np


def one_hot(x, vocab_size, dtype='float32'):
    shape = len(x), vocab_size
    r = np.zeros(shape).astype(dtype)
    for i, n in enumerate(x):
        r[i, n] = 1
    return r
