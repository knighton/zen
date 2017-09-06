from ... import api as Z
from ..layer import Layer, Spec, Sugar


class TanhShrinkLayer(Layer):
    def forward(self, x, is_training):
        return Z.tanh_shrink(x)


class TanhShrinkSpec(Spec):
    def build(self, in_shape, in_dtype):
        return TanhShrinkLayer(), in_shape, in_dtype


TanhShrink = Sugar(TanhShrinkSpec)
