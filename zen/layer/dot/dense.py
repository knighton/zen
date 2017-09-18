from ... import func as Z
from ... import init
from ..base import Transform, TransformSpec, Sugar


class DenseLayer(Transform):
    def __init__(self, kernel, bias):
        super().__init__()
        self.kernel = self.add_param(kernel)
        self.bias = self.add_param(bias)

    def forward(self, x, is_training):
        return Z.matmul(x, self.kernel) + self.bias


class DenseSpec(TransformSpec):
    def __init__(self, channels=None, kernel_init='glorot_uniform',
                 bias_init='zero'):
        super().__init__()
        if channels is not None:
            Z.check_dim(channels)
        self.channels = channels
        self.kernel_init = init.get(kernel_init)
        self.bias_init = init.get(bias_init)

    def build(self, in_shape, in_dtype):
        in_channels, = in_shape
        out_channels = self.channels
        kernel_shape = in_channels, out_channels
        kernel = self.kernel_init(kernel_shape, in_dtype)
        bias_shape = out_channels,
        bias = self.bias_init(bias_shape, in_dtype)
        out_shape = out_channels,
        return DenseLayer(kernel, bias), out_shape, in_dtype


Dense = Sugar(DenseSpec)
