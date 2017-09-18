from ... import func as Z
from ..base import Transform, TransformSpec, Sugar


class PoolLayer(Transform):
    def __init__(self, window, padding, stride):
        super().__init__()
        self.window = window
        self.padding = padding
        self.stride = stride


class PoolSpec(TransformSpec):
    """
    Pooling spec abstract base class.
    """

    def __init__(self, window=2, padding=0, stride=None, ndim=None):
        """
        window   {dim, shape}        Int means repeat per dimension.
        padding  {dim, shape}        Padding.
        stride   {None, dim, shape}  None means match window.
                                     Int means repeat per dimension.
        ndim     {None, 1, 2, 3}     Optinally specify dimensionality of input.
        """
        super().__init__()
        Z.check_input_ndim(ndim, {1, 2, 3}, 'ndim')
        Z.check_dim_or_shape(window, ndim, 'window')
        Z.check_coord_or_coords(padding, ndim, 'padding')
        if stride is None:
            stride = window
        else:
            Z.check_dim_or_shape(stride, ndim)
        self.window = window
        self.padding = padding
        self.stride = stride
        self.ndim = ndim

    def make_layer(self, ndim):
        raise NotImplementedError

    def build(self, in_shape, in_dtype):
        ndim = Z.verify_input_ndim(self.ndim, in_shape)
        out_shape = (in_shape[0],) + Z.pool_out_shape(
            in_shape[1:], self.window, self.padding, self.stride)
        return self.make_layer(ndim), out_shape, in_dtype


class AvgPoolLayer(PoolLayer):
    """
    Average pooling layer.
    """

    def __init__(self, ndim, window, padding, stride):
        super().__init__(window, padding, stride)
        self.avg_pool = Z.get('avg_pool', ndim)

    def forward(self, x, is_training):
        return self.avg_pool(x, self.window, self.padding, self.stride)


class AvgPoolSpec(PoolSpec):
    """
    Average pooling spec.
    """

    def make_layer(self, ndim, window, padding, stride):
        return AvgPoolLayer(self.ndim, self.window, self.padding, self.stride)


AvgPool = Sugar(AvgPoolSpec)
AvgPool1D = Sugar(AvgPoolSpec, {'ndim': 1})
AvgPool2D = Sugar(AvgPoolSpec, {'ndim': 2})
AvgPool3D = Sugar(AvgPoolSpec, {'ndim': 3})


class MaxPoolLayer(PoolLayer):
    """
    Max pooling layer.
    """

    def __init__(self, ndim, window, padding, stride):
        super().__init__(window, padding, stride)
        self.max_pool = Z.get('max_pool', ndim)

    def forward(self, x, is_training):
        return self.max_pool(x, self.window, self.padding, self.stride)


class MaxPoolSpec(PoolSpec):
    """
    Max pooling spec.
    """

    def make_layer(self, ndim):
        return MaxPoolLayer(self.ndim, self.window, self.padding, self.stride)


MaxPool = Sugar(MaxPoolSpec)
MaxPool1D = Sugar(MaxPoolSpec, {'ndim': 1})
MaxPool2D = Sugar(MaxPoolSpec, {'ndim': 2})
MaxPool3D = Sugar(MaxPoolSpec, {'ndim': 3})
