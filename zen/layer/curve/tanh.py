from ... import func as Z
from ..base import Transform, TransformSpec, Sugar


class TanhLayer(Transform):
    def forward(self, x, is_training):
        return Z.tanh(x)


class TanhSpec(TransformSpec):
    def build(self, in_shape, in_dtype):
        return TanhLayer(), in_shape, in_dtype


Tanh = Sugar(TanhSpec)
