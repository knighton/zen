from ... import api as Z
from ..layer import Layer, Spec, Sugar


class HardSigmoidLayer(Layer):
    def forward(self, x, is_training):
        return Z.hard_sigmoid(x)


class HardSigmoidSpec(Spec):
    def build(self, in_shape, in_dtype):
        return HardSigmoidLayer(), in_shape, in_dtype


HardSigmoid = Sugar(HardSigmoidSpec)
