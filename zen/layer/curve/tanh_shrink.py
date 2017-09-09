from ... import api as Z
from ..base import Transform, TransformSpec, Sugar


class TanhShrinkLayer(Transform):
    def forward(self, x, is_training):
        return Z.tanh_shrink(x)


class TanhShrinkSpec(TransformSpec):
    def build(self, in_shape, in_dtype):
        return TanhShrinkLayer(), in_shape, in_dtype


TanhShrink = Sugar(TanhShrinkSpec)
