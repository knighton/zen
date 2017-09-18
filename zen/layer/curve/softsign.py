from ... import func as Z
from ..base import Transform, TransformSpec, Sugar


class SoftSignLayer(Transform):
    def forward(self, x, is_training):
        return Z.softsign(x)


class SoftSignSpec(TransformSpec):
    def build(self, in_shape, in_dtype):
        return SoftSignLayer(), in_shape, in_dtype


SoftSign = Sugar(SoftSignSpec)
