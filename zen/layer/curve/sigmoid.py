from ... import api as Z
from ..layer import Layer, Spec, Sugar


class SigmoidLayer(Layer):
    def forward(self, x, is_training):
        return Z.sigmoid(x)


class SigmoidSpec(Spec):
    def build(self, in_shape, in_dtype):
        return SigmoidLayer(), in_shape, in_dtype


Sigmoid = Sugar(SigmoidSpec)
