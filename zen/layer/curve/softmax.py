from ... import functional as F
from ..layer import Layer, Spec, Sugar


class SoftmaxLayer(Layer):
    def forward(self, x, is_training):
        return F.softmax(x)


class SoftmaxSpec(Spec):
    def build(self, in_shape, in_dtype):
        return SoftmaxLayer(), in_shape, in_dtype


Softmax = Sugar(SoftmaxSpec)
