from ... import api as Z
from ... import init
from ..layer import Layer, Spec, Sugar


class ConvBaseLayer(Layer):
    def __init__(self, kernel, bias, padding, stride, dilation):
        super().__init__()
        self.kernel = self.add_param(kernel)
        self.bias = self.add_param(bias)
        self.padding = padding
        self.stride = stride
        self.dilation = dilation


class ConvBaseSpec(Spec):
    def __init__(self, channels=None, window=3, padding=1, stride=1, dilation=1,
                 kernel_init='glorot_uniform', bias_init='zero', dim=None):
        """
        channels     {None, dim}         Output channels.
                                         None means match input channels.
        window       {dim, shape}        Kernel shape without channel dims.
                                         Int means repeat per dimension.
        padding      {dim, shape}        Padding (single ints are repeated).
        stride       {dim, shape}        Stride (single ints are repeated).
        dilation     {dim, shape}        Dilation (single ints are repeated).
        kernel_init  {str, Initializer}  Kernel initializer.
        bias_init    {str, Initializer}  Bias initializer.
        dim          {None, 1, 2, 3}     Specifies input dimensionality.
        """
        super().__init__()
        if channels is not None:
            assert Z.is_dim(channels)
        assert dim in {None, 1, 2, 3}
        assert Z.is_shape_or_one(window, dim)
        assert Z.is_shape_or_one(padding, dim)
        assert Z.is_shape_or_one(stride, dim)
        assert Z.is_shape_or_one(dilation, dim)
        self.channels = channels
        self.window = window
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.kernel_init = init.get(kernel_init)
        self.bias_init = init.get(bias_init)
        self.dim = dim

    def make_layer(self, kernel, bias, padding, stride, dilation):
        raise NotImplementedError

    def compute_out_shape(self, in_shape, window, padding, stride, dilation):
        raise NotImplementedError

    def build(self, in_shape, in_dtype):
        if self.channels is None:
            out_channels = in_shape[0]
        else:
            out_channels = self.channels
        if self.dim is None:
            dim = len(in_shape) - 1
        else:
            dim = self.dim
            assert len(in_shape) == dim + 1
        in_channels = in_shape[0]
        window = Z.to_shape(self.window, dim)
        kernel_shape = (out_channels, in_channels) + window
        kernel = self.kernel_init(kernel_shape, in_dtype)
        bias_shape = out_channels,
        bias = self.bias_init(bias_shape, in_dtype)
        layer = self.make_layer(kernel, bias, self.padding, self.stride,
                                self.dilation)
        out_shape = (out_channels,) + self.compute_out_shape(
            in_shape[1:], window, self.padding, self.stride, self.dilation)
        return layer, out_shape, in_dtype


class ConvLayer(ConvBaseLayer):
    def __init__(self, kernel, bias, padding, stride, dilation):
        super().__init__(kernel, bias, padding, stride, dilation)
        dim = Z.shape(kernel) - 2
        self.conv = Z.get('conv', dim)

    def forward(self, x, is_training):
        return self.conv(x, self.kernel, self.bias, self.padding, self.stride,
                         self.dilation)


class ConvSpec(ConvBaseSpec):
    def make_layer(self, kernel, bias, padding, stride, dilation):
        return ConvLayer(kernel, bias, padding, stride, dilation)

    def compute_out_shape(self, in_shape, window, padding, stride, dilation):
        return Z.conv_out_shape(in_shape, window, padding, stride, dilation)


Conv = Sugar(ConvSpec)
Conv1D = Sugar(ConvSpec, {'dim': 1})
Conv2D = Sugar(ConvSpec, {'dim': 2})
Conv3D = Sugar(ConvSpec, {'dim': 3})
