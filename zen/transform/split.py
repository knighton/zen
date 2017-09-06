from time import time
from tqdm import tqdm

from .transform import Transform


class Split(Transform):
    def transform(self, x, verbose=0, depth=0):
        t0 = time()
        rrr = []
        if verbose == 2:
            x = tqdm(x, leave=False)
        for line in x:
            if isinstance(line, list):
                line = ''.join(line)
            tokens = line.split()
            rrr.append(tokens)
        t = time() - t0
        self.done(t, verbose, depth)
        return rrr

    def inverse_transform(x):
        rrr = []
        for line in x:
            rrr.append(' '.join(line))
        return rrr
