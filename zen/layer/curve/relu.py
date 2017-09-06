from ... import api as Z
from ..layer import Layer, Spec, Sugar


class ReLULayer(Layer):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x, is_training):
        return Z.relu(x, self.low, self.high)


class ReLUSpec(Spec):
    def __init__(self, low=0., high=None):
        super().__init__()
        self.low = low
        self.high = high

    def build(self, in_shape, in_dtype):
        return ReLULayer(self.low, self.high), in_shape, in_dtype


ReLU = Sugar(ReLUSpec)
ReLU6 = Sugar(ReLUSpec, {'high': 6.})
Threshold = Sugar(ReLUSpec, {'low': -1.})
