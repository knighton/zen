import numpy as np
from time import time

from .transform import Transform


class YesNo(Transform):
    def transform(self, x, verbose=0, depth=0):
        t0 = time()
        nn = []
        for s in x:
            n = {
                'yes': 1,
                'no': 0,
            }[s]
            nn.append(n)
        ret = np.array(nn)
        t = time() - t0
        self.done(t, verbose, depth)
        return ret

    def inverse_transform(self, x):
        ss = []
        for n in x:
            s = {
                1: 'yes',
                0: 'no',
            }[n]
            ss.append(s)
        return ss
