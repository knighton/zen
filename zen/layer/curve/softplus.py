from ... import api as Z
from ..base import Transform, TransformSpec, Sugar


class SoftPlusLayer(Transform):
    def forward(self, x, is_training):
        return Z.softplus(x)


class SoftPlusSpec(TransformSpec):
    def build(self, in_shape, in_dtype):
        return SoftPlusLayer(), in_shape, in_dtype


SoftPlus = Sugar(SoftPlusSpec)
