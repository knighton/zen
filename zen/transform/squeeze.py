import numpy as np
from time import time

from .transform import Transform


class Squeeze(Transform):
    def __init__(self, axis):
        self.axis = axis

    def transform(self, x, verbose=0, depth=0):
        t0 = time()
        ret = np.sqeeuze(x, self.axis)
        t = time() - t0
        self.done(t, verbose, depth)
        return ret

    def inverse_transform(self, x):
        return np.expand_dims(x, self.axis)
