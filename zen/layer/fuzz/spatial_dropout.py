from ... import functional as F
from ..layer import Layer, Spec, Sugar


class SpatialDropoutLayer(Layer):
    def __init__(self, dim, rate):
        super().__init__()
        self.spatial_dropout = F.get('spatial_dropout', dim)
        self.rate = rate

    def forward(self, x, is_training):
        return self.spatial_dropout(x, is_training, self.rate)


class SpatialDropoutSpec(Spec):
    def __init__(self, rate, dim=None):
        super().__init__()
        assert 0. < rate < 1.
        self.rate = rate
        assert dim in {None, 1, 2, 3}
        self.dim = dim

    def build(self, in_shape, in_dtype):
        if self.dim is None:
            dim = len(in_shape) - 1
        else:
            assert F.is_shape(in_shape, self.dim + 1)
            dim = self.dim
        return SpatialDropoutLayer(dim, self.rate), in_shape, in_dtype


SpatialDropout = Sugar(SpatialDropoutSpec)
SpatialDropout1D = Sugar(SpatialDropoutSpec, {'dim': 1})
SpatialDropout2D = Sugar(SpatialDropoutSpec, {'dim': 2})
SpatialDropout3D = Sugar(SpatialDropoutSpec, {'dim': 3})
