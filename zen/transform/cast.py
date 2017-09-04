from .transform import Transform


class Cast(Transform):
    def __init__(self, dtype):
        self.dtype = dtype

    def transform(self, x):
        return x.astype(self.dtype)

    def inverse_transform(self, x):
        return x
