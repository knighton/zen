from torch.nn import functional as F

from ..core.util import to_one, to_shape


def conv1d(x, kernel, bias, padding, stride, dilation):
    """
    x         tensor (batch_size, in_channels, width)
    kernel    tensor (out_channels, in_channels, width)
    bias      tensor (out_channels,)
    padding   dim or shape
    stride    dim or shape
    dilation  dim or shape
    """
    padding = to_one(padding)
    stride = to_one(stride)
    dilation = to_one(dilation)
    return F.conv1d(x, kernel, bias, stride, padding, dilation, 1)


def conv2d(x, kernel, bias, padding, stride, dilation):
    """
    x         tensor (batch_size, in_channels, height, width)
    kernel    tensor (out_channels, in_channels, height, width)
    bias      tensor (out_channels,)
    padding   dim or shape
    stride    dim or shape
    dilation  dim or shape
    """
    return F.conv2d(x, kernel, bias, stride, padding, dilation, 1)


def conv3d(x, kernel, bias, padding, stride, dilation):
    """
    x         tensor (batch_size, in_channels, depth, height, width)
    kernel    tensor (out_channels, in_channels, depth, height, width)
    bias      tensor (out_channels,)
    padding   dim or shape
    stride    dim or shape
    dilation  dim or shape
    """
    return F.conv3d(x, kernel, bias, stride, padding, dilation, 1)


def conv_out_shape(in_shape, filter_shape, padding, stride, dilation):
    """
    in_shape      shape... (of x without channel dim)
    filter_shape  dim or shape
    padding       dim or shape
    stride        dim or shape
    dilation      dim or shape
    """
    dim = len(in_shape)
    filter_shape = to_shape(filter_shape, dim)
    padding = to_shape(padding, dim)
    stride = to_shape(stride, dim)
    dilation = to_shape(dilation, dim)
    out_shape = [None] * len(in_shape)
    for i in range(len(in_shape)):
        numerator = in_shape[i] + 2 * padding[i] - \
                    dilation[i] * (filter_shape[i] - 1) - 1
        out_shape[i] = floor(numerator / stride[i] + 1)
    return tuple(out_shape)
