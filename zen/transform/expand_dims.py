import numpy as np

from .transform import Transform


class ExpandDims(Transform):
    def __init__(self, axis):
        self.axis = axis

    def transform(self, x):
        return np.expand_dims(x, self.axis)

    def inverse_transform(self, x):
        return np.squeeze(x, self.axis)
