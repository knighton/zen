import mxnet as mx

from ....core.util import to_shape


def conv(x, kernel, bias, padding, stride, dilation):
    ndim = x.ndim - 2
    padding = to_shape(padding, ndim)
    stride = to_shape(stride, ndim)
    dilation = to_shape(dilation, ndim)
    return mx.nd.Convolution(
        x, kernel, bias, kernel.shape[2:], stride, dilation, padding,
        kernel.shape[0])


conv1d = conv
conv2d = conv
conv3d = conv
