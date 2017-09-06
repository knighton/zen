from time import time
from tqdm import tqdm

from .transform import Transform


class Length(Transform):
    pad = None

    def __init__(self, length):
        self.length = length

    def transform(self, x, verbose=0, depth=0):
        t0 = time()
        rrr = []
        pre = []
        if verbose == 2:
            x = tqdm(x, leave=False)
        for line in x:
            if isinstance(line, str):
                line = list(line)
            pre.append(len(line))
            if len(line) < self.length:
                rr = line + [self.pad] * (self.length - len(line))
            else:
                rr = line[:self.length]
            rrr.append(rr)
        pre.sort()
        tiles = []
        for f in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
            x = int(len(pre) * f)
            if x == len(pre):
                x -= 1
            tiles.append(pre[x])
        print('Length percentiles: %s' % tiles)
        t = time() - t0
        self.done(t, verbose, depth)
        return rrr

    def inverse_transform(self, x):
        rrr = []
        for line in x:
            for i in range(len(x)):
                token = x[x - 1]
                if token != self.pad:
                    rr = x[:-i]
                    break
            else:
                rr = []
            rrr.append(rr)
        return rrr
