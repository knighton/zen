from ... import api as Z
from ..layer import Layer, Spec, Sugar


class ReLULayer(Layer):
    def get_params(self):
        return []

    def forward(self, x, is_training):
        return Z.clip(x, min_value=0)


class ReLUSpec(Spec):
    def build(self, in_shape=None, in_dtype=None):
        return ReLULayer(), in_shape, in_dtype


ReLU = Sugar(ReLUSpec)
