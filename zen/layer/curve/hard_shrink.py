from ... import func as Z
from ..base import Transform, TransformSpec, Sugar


class HardShrinkLayer(Transform):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x, is_training):
        return Z.hard_shrink(x)


class HardShrinkSpec(TransformSpec):
    def __init__(self, lambda_=0.5):
        super().__init__()
        self.lambda_ = lambda_

    def build(self, in_shape, in_dtype):
        return HardShrinkLayer(self.lambda_), in_shape, in_dtype


HardShrink = Sugar(HardShrinkSpec)
