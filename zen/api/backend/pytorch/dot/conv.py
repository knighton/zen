from torch.nn import functional as F

from ... import to_one


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
