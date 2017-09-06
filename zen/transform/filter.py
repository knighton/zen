from time import time
from tqdm import tqdm

from .transform import Transform


class Filter(Transform):
    def __init__(self, filter_):
        self.filter = filter_

    def transform(self, x, verbose=0, depth=0):
        t0 = time()
        rrr = []
        if verbose == 2:
            x = tqdm(x, leave=False)
        for line in x:
            rr = list(filter(lambda token: token in self.filter, line))
            rrr.append(rr)
        t = time() - t0
        self.done(t, verbose, depth)
        return rrr

    def inverse_transform(x):
        return x
