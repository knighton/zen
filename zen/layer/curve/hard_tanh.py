from ... import api as Z
from ..layer import Layer, Spec, Sugar


class HardTanhLayer(Layer):
    def forward(self, x, is_training):
        return Z.hard_tanh(x)


class HardTanhSpec(Spec):
    def build(self, in_shape, in_dtype):
        return HardTanhLayer(), in_shape, in_dtype


HardTanh = Sugar(HardTanhSpec)
