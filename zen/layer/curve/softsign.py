from ... import api as Z
from ..layer import Layer, Spec, Sugar


class SoftSignLayer(Layer):
    def forward(self, x, is_training):
        return Z.softsign(x)


class SoftSignSpec(Spec):
    def build(self, in_shape, in_dtype):
        return SoftSignLayer(), in_shape, in_dtype


SoftSign = Sugar(SoftSignSpec)
