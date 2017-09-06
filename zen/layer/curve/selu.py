from ... import api as Z
from ..layer import Layer, Spec, Sugar


class SELULayer(Layer):
    def forward(self, x, is_training):
        return Z.selu(x)


class SELUSpec(Spec):
    def build(self, in_shape, in_dtype):
        return SELULayer(), in_shape, in_dtype


SELU = Sugar(SELUSpec)
