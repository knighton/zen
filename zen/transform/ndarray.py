import numpy as np
from time import time

from .transform import Transform


class NDArray(Transform):
    def __init__(self, dtype):
        self.dtype = dtype

    def transform(self, x, verbose=0, depth=0):
        t0 = time()
        ret = np.array(x).astype(self.dtype)
        t = time() - t0
        self.done(t, verbose, depth)
        return ret

    def inverse_transform(self, x, verbose, depth):
        return x
