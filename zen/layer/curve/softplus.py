from ... import api as Z
from ..layer import Layer, Spec, Sugar


class SoftPlusLayer(Layer):
    def forward(self, x, is_training):
        return Z.softplus(x)


class SoftPlusSpec(Spec):
    def build(self, in_shape, in_dtype):
        return SoftPlusLayer(), in_shape, in_dtype


SoftPlus = Sugar(SoftPlusSpec)
