import numpy as np

from .transform import Transform


class NDArray(Transform):
    def __init__(self, dtype):
        self.dtype = dtype

    def transform(self, x):
        return np.array(x).astype(self.dtype)

    def inverse_transform(self, x):
        return x.tolist()
