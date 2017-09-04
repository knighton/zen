import numpy as np

from .transform import Transform


class Squeeze(Transform):
    def __init__(self, axis):
        self.axis = axis

    def transform(self, x):
        return np.sqeeuze(x, self.axis)

    def inverse_transform(self, x):
        return np.expand_dims(x, self.axis)
