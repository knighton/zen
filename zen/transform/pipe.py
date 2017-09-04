import inspect

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

    def fit(self, x):
        for step in self.steps:
            x = step.fit_transform(x)

    def transform(self, x):
        for step in self.steps:
            x = step.transform(x)
        return x

    def fit_transform(self, x):
        for step in self.steps:
            x = step.fit_transform(x)
        return x

    def inverse_transform(self, x):
        for step in reversed(self.steps):
            x = step.inverse_transform(x)
        return x
