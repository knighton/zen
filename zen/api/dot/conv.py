from math import floor

from .. import backend as Z
from ..util import to_shape


"""
1D convolution.  See also `conv_out_shape`.

Input:
    x         variable (batch_size, in_channels, width)
    kernel    variable (out_channels, in_channels, width)
    bias      variable (out_channels,)
    padding   dim or shape
    stride    dim or shape
    dilation  dim or shape

Output:
    x         variable (batch_size, out_channels, width)
"""
conv1d = Z.conv1d


"""
2D convolution.  See also `conv_out_shape`.

Input:
    x         variable (batch_size, in_channels, height, width)
    kernel    variable (out_channels, in_channels, height, width)
    bias      variable (out_channels,)
    padding   dim or shape
    stride    dim or shape
    dilation  dim or shape

Output:
    x         variable (batch_size, out_channels, height, width)
"""
conv2d = Z.conv2d


"""
3D convolution.  See also `conv_out_shape`.

Input:
    x         variable (batch_size, in_channels, depth, height, width)
    kernel    variable (out_channels, in_channels, depth, height, width)
    bias      variable (out_channels,)
    padding   dim or shape
    stride    dim or shape
    dilation  dim or shape

Output:
    x         variable (batch_size, out_channels, depth, height, width)
"""
conv3d = Z.conv3d


_DIM2CONV = {
    1: conv1d,
    2: conv2d,
    3: conv3d,
}


def conv(x, is_training, rate):
    """
    ND convolution.  See also `conv_out_shape`.

    Input:
        x         variable (batch_size, in_channels, shape...)
        kernel    variable (out_channels, in_channels, shape...)
        bias      variable (out_channels,)
        padding   dim or shape
        stride    dim or shape
        dilation  dim or shape

    Output:
        x         variable (batch_size, out_channels, shape...)
    """
    dim = x.size() - 2
    return _DIM2CONV[dim](x, is_training, rate)


def conv_out_shape(in_shape, window, padding, stride, dilation):
    """
    Calculate convolution output shape.

    Input:
        in_shape      shape         The shape of x without channel dimension.
        window        dim or shape  Filter shape.
        padding       dim or shape  Padding.
        stride        dim or shape  Stride.
        dilation      dim or shape  Dilation.

    Output:
        out_shape     shape         The shape of y without channel dimension.
    """
    dim = len(in_shape)
    window = to_shape(window, dim)
    padding = to_shape(padding, dim)
    stride = to_shape(stride, dim)
    dilation = to_shape(dilation, dim)
    out_shape = [None] * len(in_shape)
    for i in range(len(in_shape)):
        numerator = \
            in_shape[i] + 2 * padding[i] - dilation[i] * (window[i] - 1) - 1
        out_shape[i] = floor(numerator / stride[i] + 1)
    return tuple(out_shape)
