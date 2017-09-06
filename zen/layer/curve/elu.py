from ... import api as Z
from ..layer import Layer, Spec, Sugar


class ELULayer(Layer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, is_training):
        return Z.elu(x, self.alpha)


class ELUSpec(Spec):
    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def build(self, in_shape, in_dtype):
        return ELULayer(self.alpha), in_shape, in_dtype


ELU = Sugar(ELUSpec)
