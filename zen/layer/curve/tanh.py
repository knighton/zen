from ... import api as Z
from ..layer import Layer, Spec, Sugar


class TanhLayer(Layer):
    def forward(self, x, is_training):
        return Z.tanh(x)


class TanhSpec(Spec):
    def build(self, in_shape, in_dtype):
        return TanhLayer(), in_shape, in_dtype


Tanh = Sugar(TanhSpec)
