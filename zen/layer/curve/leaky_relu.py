from ... import api as Z
from ..layer import Layer, Spec, Sugar


class LeakyReLULayer(Layer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, is_training):
        return Z.leaky_relu(x, self.alpha)


class LeakyReLUSpec(Spec):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def build(self, in_shape, in_dtype):
        return LeakyReLULayer(self.alpha), in_shape, in_dtype


LeakyReLU = Sugar(LeakyReLUSpec)
