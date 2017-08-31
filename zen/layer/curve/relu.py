from ..layer import Layer, Spec, Sugar


class ReLULayer(Layer):
    def get_params(self):
        return []

    def forward(self, x, is_training):
        return x.clamp(min=0)


class ReLUSpec(Spec):
    def build(self, in_shape=None, in_dtype=None):
        return ReLULayer(), in_shape, in_dtype


ReLU = Sugar(ReLUSpec)
