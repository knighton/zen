from ... import api as Z
from ..layer import Layer, Spec, Sugar


class SpatialDropoutLayer(Layer):
    def __init__(self, ndim, rate):
        super().__init__()
        self.spatial_dropout = Z.get('spatial_dropout', ndim)
        self.rate = rate

    def forward(self, x, is_training):
        return self.spatial_dropout(x, is_training, self.rate)


class SpatialDropoutSpec(Spec):
    def __init__(self, rate, ndim=None):
        super().__init__()
        assert 0. < rate < 1.
        self.rate = rate
        assert ndim in {None, 1, 2, 3}
        self.ndim = ndim

    def build(self, in_shape, in_dtype):
        if self.ndim is None:
            ndim = len(in_shape) - 1
        else:
            assert Z.is_shape(in_shape, self.ndim + 1)
            ndim = self.ndim
        return SpatialDropoutLayer(ndim, self.rate), in_shape, in_dtype


SpatialDropout = Sugar(SpatialDropoutSpec)
SpatialDropout1D = Sugar(SpatialDropoutSpec, {'ndim': 1})
SpatialDropout2D = Sugar(SpatialDropoutSpec, {'ndim': 2})
SpatialDropout3D = Sugar(SpatialDropoutSpec, {'ndim': 3})
