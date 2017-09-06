from time import time
from tqdm import tqdm

from .transform import Transform


class Lower(Transform):
    def transform(self, x, verbose=0, depth=0):
        t0 = time()
        rrr = []
        if verbose == 2:
            x = tqdm(x, leave=False)
        for line in x:
            if isinstance(line, str):
                rr = line.lower()
            else:
                rr = []
                for token in line:
                    rr.append(token.lower())
            rrr.append(rr)
        t = time() - t0
        self.done(t, verbose, depth)
        return rrr

    def inverse_transform(x):
        return x
