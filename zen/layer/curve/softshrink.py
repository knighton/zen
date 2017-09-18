from ... import func as Z
from ..base import Transform, TransformSpec, Sugar


class SoftShrinkLayer(Transform):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x, is_training):
        return Z.softshrink(x, self.lambda_)


class SoftShrinkSpec(TransformSpec):
    def __init__(self, lambda_=0.5):
        super().__init__()
        self.lambda_ = lambda_

    def build(self, in_shape, in_dtype):
        return SoftShrinkLayer(self.lambda_), in_shape, in_dtype


SoftShrink = Sugar(SoftShrinkSpec)
