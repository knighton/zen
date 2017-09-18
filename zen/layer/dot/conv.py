from ... import func as Z
from ... import init
from ..base import Transform, TransformSpec, Sugar


class ConvBaseLayer(Transform):
    def __init__(self, kernel, bias, padding, stride, dilation):
        super().__init__()
        self.kernel = self.add_param(kernel)
        self.bias = self.add_param(bias)
        self.padding = padding
        self.stride = stride
        self.dilation = dilation


class ConvBaseSpec(TransformSpec):
    def __init__(self, channels=None, window=3, padding=1, stride=1, dilation=1,
                 kernel_init='glorot_uniform', bias_init='zero', ndim=None):
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
        ndim         {None, 1, 2, 3}     Specifies input dimensionality.
        """
        super().__init__()
        Z.check_out_channels(channels, 'channels')
        Z.check_input_ndim(ndim, {1, 2, 3}, 'ndim')
        Z.check_dim_or_shape(window, ndim, 'window'),
        Z.check_coord_or_coords(padding, ndim, 'padding')
        Z.check_dim_or_shape(stride, ndim, 'stride')
        Z.check_dim_or_shape(dilation, ndim, 'dilation')
        self.channels = channels
        self.window = window
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.kernel_init = init.get(kernel_init)
        self.bias_init = init.get(bias_init)
        self.ndim = ndim

    def make_layer(self, kernel, bias, padding, stride, dilation):
        raise NotImplementedError

    def compute_out_shape(self, in_shape, window, padding, stride, dilation):
        raise NotImplementedError

    def build(self, in_shape, in_dtype):
        if self.channels is None:
            out_channels = in_shape[0]
        else:
            out_channels = self.channels
        ndim = Z.verify_input_ndim(self.ndim, in_shape)
        in_channels = in_shape[0]
        window = Z.to_shape(self.window, ndim)
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
        ndim = Z.ndim(self.kernel) - 2
        self.conv = Z.get('conv', ndim)

    def forward(self, x, is_training):
        return self.conv(x, self.kernel, self.bias, self.padding, self.stride,
                         self.dilation)


class ConvSpec(ConvBaseSpec):
    def make_layer(self, kernel, bias, padding, stride, dilation):
        return ConvLayer(kernel, bias, padding, stride, dilation)

    def compute_out_shape(self, in_shape, window, padding, stride, dilation):
        return Z.conv_out_shape(in_shape, window, padding, stride, dilation)


Conv = Sugar(ConvSpec)
Conv1D = Sugar(ConvSpec, {'ndim': 1})
Conv2D = Sugar(ConvSpec, {'ndim': 2})
Conv3D = Sugar(ConvSpec, {'ndim': 3})
