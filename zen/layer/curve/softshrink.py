from ... import api as Z
from ..layer import Layer, Spec, Sugar


class SoftShrinkLayer(Layer):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x, is_training):
        return Z.softshrink(x, self.lambda_)


class SoftShrinkSpec(Spec):
    def __init__(self, lambda_=0.5):
        super().__init__()
        self.lambda_ = lambda_

    def build(self, in_shape, in_dtype):
        return SoftShrinkLayer(self.lambda_), in_shape, in_dtype


SoftShrink = Sugar(SoftShrinkSpec)
