import numpy as np

from .transform import Transform


class YesNo(Transform):
    def transform(self, x):
        nn = []
        for s in x:
            n = {
                'yes': 1,
                'no': 0,
            }[s]
            nn.append(n)
        return np.array(nn)

    def inverse_transform(self, x):
        ss = []
        for n in x:
            s = {
                1: 'yes',
                0: 'no',
            }[n]
            ss.append(s)
        return ss
