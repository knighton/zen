from ... import func as Z
from ..base import Transform, TransformSpec, Sugar


class SoftmaxLayer(Transform):
    def forward(self, x, is_training):
        return Z.softmax(x)


class SoftmaxSpec(TransformSpec):
    def build(self, in_shape, in_dtype):
        return SoftmaxLayer(), in_shape, in_dtype


Softmax = Sugar(SoftmaxSpec)
