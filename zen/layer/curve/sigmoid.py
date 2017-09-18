from ... import func as Z
from ..base import Transform, TransformSpec, Sugar


class SigmoidLayer(Transform):
    def forward(self, x, is_training):
        return Z.sigmoid(x)


class SigmoidSpec(TransformSpec):
    def build(self, in_shape, in_dtype):
        return SigmoidLayer(), in_shape, in_dtype


Sigmoid = Sugar(SigmoidSpec)
