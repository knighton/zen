import numpy as np

from .transform import Transform


class TrueFalse(Transform):
    def transform(self, x, verbose=0, depth=0):
        nn = []
        for s in x:
            n = {
                'true': 1,
                'false': 0,
            }[s]
            nn.append(n)
        return np.array(nn)

    def inverse_transform(self, x):
        ss = []
        for n in x:
            s = {
                1: 'true',
                0: 'false',
            }[n]
            ss.append(s)
        return ss
