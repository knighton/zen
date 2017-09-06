import inspect
from time import time

from .transform import Transform


class Pipe(Transform):
    def __init__(self, *steps):
        self.steps = []
        for step in steps:
            if isinstance(step, Transform):
                pass
            elif inspect.isclass(step) and issubclass(step, Transform):
                step = step()
            else:
                assert False
            self.steps.append(step)

    def fit(self, x, verbose=0, depth=0):
        self.start_pipe(verbose, depth)
        t0 = time()
        for step in self.steps:
            x = step.fit_transform(x, verbose, depth + 1)
        t = time() - t0
        self.done(t, verbose, depth)

    def transform(self, x, verbose=0, depth=0):
        self.start_pipe(verbose, depth)
        t0 = time()
        for step in self.steps:
            x = step.transform(x, verbose, depth + 1)
        t = time() - t0
        self.done(t, verbose, depth)
        return x

    def fit_transform(self, x, verbose=0, depth=0):
        self.start_pipe(verbose, depth)
        t0 = time()
        for step in self.steps:
            x = step.fit_transform(x, verbose, depth + 1)
        t = time() - t0
        self.done(t, verbose, depth)
        return x

    def inverse_transform(self, x):
        for step in reversed(self.steps):
            x = step.inverse_transform(x)
        return x
