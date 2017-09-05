import numpy as np

from .. import backend as Z
from .. import core as C


"""
Constant padding.

    x, padding, value -> y

Input:
                                           1D   2D    3D
                                           --   --    --
    x        variable                      NCW  NCHW  NCDHW
    padding  {coord, coords, coord pairs}  W    HW    DHW
             for spatial dims only
    value    {int, float}

Output:
    y        variable                      NCW  NCHW  NCDHW
"""
constant_pad = Z.constant_pad
constant_pad1d = Z.constant_pad1d
constant_pad2d = Z.constant_pad2d
constant_pad3d = Z.constant_pad3d


"""
Edge padding.

    x, padding -> y

Input:
                                           1D   2D    3D
                                           --   --    --
    x        variable                      NCW  NCHW  NCDHW
    padding  {coord, coords, coord pairs}  W    HW    DHW
             for spatial dims only

Output:
    y        variable                      NCW  NCHW  NCDHW
"""
edge_pad = Z.edge_pad
edge_pad1d = Z.edge_pad1d
edge_pad2d = Z.edge_pad2d
edge_pad3d = Z.edge_pad3d


"""
Reflect padding.

    x, padding -> y

Input:
                                           1D   2D    3D
                                           --   --    --
    x        variable                      NCW  NCHW  NCDHW
    padding  {coord, coords, coord pairs}  W    HW    DHW
             for spatial dims only

Output:
    y        variable                      NCW  NCHW  NCDHW
"""
reflect_pad = Z.reflect_pad
reflect_pad1d = Z.reflect_pad1d
reflect_pad2d = Z.reflect_pad2d
reflect_pad3d = Z.reflect_pad3d


def pad_out_shape(in_shape, padding):
    """
    Compute output shape of padding.

    Input:
        in_shape   sample shape of x
        padding    {int, coords, coord pairs}

    Output:
        out_shape  sample shape of y
    """
    padding = C.normalize_int_padding(padding, len(in_shape))
    out_shape = [in_shape[0]]
    for in_dim, (pad_left, pad_right) in zip(in_shape[1:], padding):
        out_shape.append(pad_left + in_dim + pad_right)
    return tuple(out_shape)
