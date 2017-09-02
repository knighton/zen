from math import floor

from .. import backend as Z
from ..util import to_shape


conv1d = Z.conv1d
conv2d = Z.conv2d
conv3d = Z.conv3d


def conv_out_shape(in_shape, window, padding, stride, dilation):
    """
    in_shape      shape (of x without channel dim)
    window        dim or shape
    padding       dim or shape
    stride        dim or shape
    dilation      dim or shape
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
