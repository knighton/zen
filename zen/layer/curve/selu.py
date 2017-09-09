from ... import api as Z
from ..base import Transform, TransformSpec, Sugar


class SELULayer(Transform):
    def forward(self, x, is_training):
        return Z.selu(x)


class SELUSpec(TransformSpec):
    def build(self, in_shape, in_dtype):
        return SELULayer(), in_shape, in_dtype


SELU = Sugar(SELUSpec)
