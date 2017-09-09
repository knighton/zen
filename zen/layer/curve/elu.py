from ... import api as Z
from ..base import Transform, TransformSpec, Sugar


class ELULayer(Transform):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, is_training):
        return Z.elu(x, self.alpha)


class ELUSpec(TransformSpec):
    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = alpha

    def build(self, in_shape, in_dtype):
        return ELULayer(self.alpha), in_shape, in_dtype


ELU = Sugar(ELUSpec)
