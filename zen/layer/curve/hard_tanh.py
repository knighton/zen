from ... import func as Z
from ..base import Transform, TransformSpec, Sugar


class HardTanhLayer(Transform):
    def forward(self, x, is_training):
        return Z.hard_tanh(x)


class HardTanhSpec(TransformSpec):
    def build(self, in_shape, in_dtype):
        return HardTanhLayer(), in_shape, in_dtype


HardTanh = Sugar(HardTanhSpec)
