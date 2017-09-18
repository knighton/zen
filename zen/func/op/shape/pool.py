from math import floor

from ... import core as C
from ... import engine as E


"""
Average pooling.

    x, window, padding, stride -> y

Input:
                           1D   2D    3D
                           --   --    --
    x        variable      NCW  NCHW  NCDHW
    window   dim or shape  W    HW    DHW
    padding  dim or shape  W    HW    DHW
    stride   dim or shape  W    HW    DHW

Output:
    y        variable      NCW  NCHW  NCDHW
"""
avg_pool = E.avg_pool
avg_pool1d = E.avg_pool1d
avg_pool2d = E.avg_pool2d
avg_pool3d = E.avg_pool3d


"""
Max pooling.

    x, window, padding, stride -> y

Input:
                           1D   2D    3D
                           --   --    --
    x        variable      NCW  NCHW  NCDHW
    window   dim or shape  W    HW    DHW
    padding  dim or shape  W    HW    DHW
    stride   dim or shape  W    HW    DHW

Output:
    y        variable      NCW  NCHW  NCDHW
"""
max_pool = E.max_pool
max_pool1d = E.max_pool1d
max_pool2d = E.max_pool2d
max_pool3d = E.max_pool3d


def pool_out_shape(in_shape, window, padding, stride):
    """
    Compute pooling output shape.

    Input:
        in_shape   shape (of x without channel dim)
        window     dim or shape
        padding    dim or shape
        stride     dim or shape

    Output:
        out_shape  shape (of y without channel dim)
    """
    dim = len(in_shape)
    window = C.to_shape(window, dim)
    padding = C.to_shape(padding, dim)
    stride = C.to_shape(stride, dim)
    out_shape = [None] * len(in_shape)
    for i in range(len(in_shape)):
        value = (in_shape[i] + 2 * padding[i] - window[i]) / stride[i]
        out_shape[i] = floor(max(value, 0)) + 1
    return tuple(out_shape)
