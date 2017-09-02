from ... import functional as F
from ..layer import Layer, Spec, Sugar


class PoolLayer(Layer):
    def __init__(self, window, padding, stride):
        super().__init__()
        self.window = window
        self.padding = padding
        self.stride = stride


class PoolSpec(Spec):
    def __init__(self, window=2, padding=0, stride=None, dim=None):
        """
        window   {dim, shape}        Int means repeat per dimension.
        padding  {dim, shape}        Padding.
        stride   {None, dim, shape}  None means match window.
                                     Int means repeat per dimension.
        dim      {None, 1, 2, 3}     Optinally specify dimensionality of input.
        """
        super().__init__()
        assert dim in {None, 1, 2, 3}
        assert F.is_shape_or_one(window, dim)
        assert F.is_shape_or_one(padding, dim)
        if stride is None:
            stride = window
        else:
            assert F.is_shape_or_one(stride, dim)
        self.window = window
        self.padding = padding
        self.stride = stride
        self.dim = dim

    def make_layer(self, dim):
        raise NotImplementedError

    def build(self, in_shape, in_dtype):
        if self.dim is None:
            dim = len(in_shape) - 1
        else:
            assert F.is_shape(in_shape, self.dim + 1)
            dim = self.dim
        out_shape = (in_shape[0],) + F.pool_out_shape(
            in_shape[1:], self.window, self.padding, self.stride)
        return self.make_layer(dim), out_shape, in_dtype


class AvgPoolLayer(PoolLayer):
    def __init__(self, dim):
        self.avg_pool = F.get('avg_pool', dim)

    def forward(self, x, is_training):
        return self.avg_pool(x, self.window, self.padding, self.stride)


class AvgPoolSpec(PoolSpec):
    def make_layer(self, dim):
        return AvgPoolLayer(self.dim, self.window, self.padding, self.stride)


AvgPool = Sugar(AvgPoolSpec)
AvgPool1D = Sugar(AvgPoolSpec, {'dim': 1})
AvgPool2D = Sugar(AvgPoolSpec, {'dim': 2})
AvgPool3D = Sugar(AvgPoolSpec, {'dim': 3})


class MaxPoolLayer(PoolLayer):
    def __init__(self, dim):
        self.max_pool = F.get('max_pool', dim)

    def forward(self, x, is_training):
        return self.max_pool(x, self.window, self.padding, self.stride)


class MaxPoolSpec(PoolSpec):
    def make_layer(self, dim):
        return MaxPoolLayer(self.dim, self.window, self.padding, self.stride)


MaxPool = Sugar(MaxPoolSpec)
MaxPool1D = Sugar(MaxPoolSpec, {'dim': 1})
MaxPool2D = Sugar(MaxPoolSpec, {'dim': 2})
MaxPool3D = Sugar(MaxPoolSpec, {'dim': 3})
