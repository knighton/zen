from ... import api as Z
from ..layer import Layer, Spec, Sugar


class GlobalPoolLayer(Layer):
    pass


class GlobalPoolSpec(Spec):
    def __init__(self, ndim=None):
        """
        ndim  {None, 1, 2, 3}  None means infer from input.
        """
        super().__init__()
        assert ndim in {None, 1, 2, 3}
        self.ndim = ndim

    def build(self, in_shape, in_dtype):
        ndim = Z.verify_input_ndim(self.ndim, in_shape)
        out_shape = in_shape[0],
        return self.make_layer(ndim), out_shape, in_dtype


class GlobalAvgPoolLayer(GlobalPoolLayer):
    def __init__(self, ndim):
        self.global_avg_pool = Z.get('global_avg_pool', ndim)

    def forward(self, x, is_training):
        return self.global_avg_pool(x)


class GlobalAvgPoolSpec(GlobalPoolSpec):
    def make_layer(self, ndim):
        return GlobalAvgPoolLayer(ndim)


GlobalAvgPool = Sugar(GlobalAvgPoolSpec)
GlobalAvgPool1D = Sugar(GlobalAvgPoolSpec, {'ndim': 1})
GlobalAvgPool2D = Sugar(GlobalAvgPoolSpec, {'ndim': 2})
GlobalAvgPool3D = Sugar(GlobalAvgPoolSpec, {'ndim': 3})


class GlobalMaxPoolLayer(GlobalPoolLayer):
    def __init__(self, ndim):
        self.global_max_pool = Z.get('global_max_pool', ndim)

    def forward(self, x, is_training):
        return self.global_max_pool(x)


class GlobalMaxPoolSpec(GlobalPoolSpec):
    def make_layer(self, ndim):
        return GlobalMaxPoolLayer(ndim)


GlobalMaxPool = Sugar(GlobalMaxPoolSpec)
GlobalMaxPool1D = Sugar(GlobalMaxPoolSpec, {'ndim': 1})
GlobalMaxPool2D = Sugar(GlobalMaxPoolSpec, {'ndim': 2})
GlobalMaxPool3D = Sugar(GlobalMaxPoolSpec, {'ndim': 3})
