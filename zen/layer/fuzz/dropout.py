from ... import api as Z
from ..layer import Layer, Spec, Sugar


class DropoutLayer(Layer):
    def __init__(self, dim, rate):
        super().__init__()
        self.dropout = Z.get('dropout', dim)
        self.rate = rate

    def forward(self, x, is_training):
        return self.dropout(x, is_training, self.rate)


class DropoutSpec(Spec):
    def __init__(self, rate, dim=None):
        super().__init__()
        assert 0. < rate < 1.
        self.rate = rate
        assert dim in {None, 0, 1, 2, 3}
        self.dim = dim

    def build(self, in_shape, in_dtype):
        if self.dim is not None:
            assert Z.is_shape(in_shape, self.dim + 1)
            dim = len(in_shape) - 1
        else:
            dim = self.dim
        return DropoutLayer(dim, self.rate), in_shape, in_dtype


Dropout = Sugar(DropoutSpec)
Dropout0D = Sugar(DropoutSpec, {'dim': 0})
Dropout1D = Sugar(DropoutSpec, {'dim': 1})
Dropout2D = Sugar(DropoutSpec, {'dim': 2})
Dropout3D = Sugar(DropoutSpec, {'dim': 3})
