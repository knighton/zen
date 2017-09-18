from math import floor

from ... import base as B
from ... import engine as E


"""
Convolution.

    x, kernel, bias, padding, stride, dilation -> y

Input:
                            1D   2D    3D
                            --   --    --
    x         varibale      NCW  NCHW  NCDHW
    kernel    variable      OCW  OCHW  OCDHW
    bias      variable      O    O     O
    padding   dim or shape  W    HW    DHW
    stride    dim or shape  W    HW    DHW
    dilation  dim or shape  W    HW    DHW

Output:
    y         variable      NOW  NOHW  NODHW
"""
conv1d = E.conv1d
conv2d = E.conv2d
conv3d = E.conv3d
conv = E.conv


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
    window = B.to_shape(window, dim)
    padding = B.to_shape(padding, dim)
    stride = B.to_shape(stride, dim)
    dilation = B.to_shape(dilation, dim)
    out_shape = [None] * len(in_shape)
    for i in range(len(in_shape)):
        numerator = \
            in_shape[i] + 2 * padding[i] - dilation[i] * (window[i] - 1) - 1
        out_shape[i] = floor(numerator / stride[i] + 1)
    return tuple(out_shape)
