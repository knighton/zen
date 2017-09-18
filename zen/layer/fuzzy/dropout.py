from ... import func as Z
from ..base import Transform, TransformSpec, Sugar


class DropoutLayer(Transform):
    def __init__(self, ndim, rate):
        super().__init__()
        self.dropout = Z.get('dropout', ndim)
        self.rate = rate

    def forward(self, x, is_training):
        return self.dropout(x, is_training, self.rate)


class DropoutSpec(TransformSpec):
    def __init__(self, rate, ndim=None):
        super().__init__()
        assert 0. < rate < 1.
        self.rate = rate
        assert ndim in {None, 0, 1, 2, 3}
        self.ndim = ndim

    def build(self, in_shape, in_dtype):
        if self.ndim is not None:
            assert Z.is_shape(in_shape, self.ndim + 1)
            ndim = len(in_shape) - 1
        else:
            ndim = self.ndim
        return DropoutLayer(ndim, self.rate), in_shape, in_dtype


Dropout = Sugar(DropoutSpec)
Dropout0D = Sugar(DropoutSpec, {'ndim': 0})
Dropout1D = Sugar(DropoutSpec, {'ndim': 1})
Dropout2D = Sugar(DropoutSpec, {'ndim': 2})
Dropout3D = Sugar(DropoutSpec, {'ndim': 3})
