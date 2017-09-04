from .transform import Transform


class Filter(Transform):
    def __init__(self, filter_):
        self.filter = filter_

    def transform(self, x):
        rrr = []
        for line in x:
            rr = list(filter(lambda token: token in self.filter, line))
            rrr.append(rr)
        return rrr

    def inverse_transform(x):
        return x
