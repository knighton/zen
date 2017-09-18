from ... import func as Z
from ..base import Transform, TransformSpec, Sugar


class HardSigmoidLayer(Transform):
    def forward(self, x, is_training):
        return Z.hard_sigmoid(x)


class HardSigmoidSpec(TransformSpec):
    def build(self, in_shape, in_dtype):
        return HardSigmoidLayer(), in_shape, in_dtype


HardSigmoid = Sugar(HardSigmoidSpec)
