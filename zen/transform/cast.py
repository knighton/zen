from time import time

from .transform import Transform


class Cast(Transform):
    def __init__(self, dtype):
        self.dtype = dtype

    def transform(self, x, verbose=0, depth=0):
        t0 = time()
        ret = x.astype(self.dtype)
        t = time() - t0
        self.done(t, verbose, depth)
        return ret

    def inverse_transform(self, x):
        return x
