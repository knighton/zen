import numpy as np

from .. import backend as Z
from .. import core as C


def _my_spatial_dropout(x, is_training, rate):
    if not is_training:
        return x
    shape = C.shape(x)
    noise_shape = shape[:2] + (1,) * (len(shape) - 2)
    max_value = 1. / rate
    mask = np.random.uniform(0, max_value, noise_shape).astype('float32')
    mask = np.floor(mask.clip(0., 1.))
    mask = C.constant(mask)
    return x * mask / (1. - rate)


"""
ND spatial dropout.

Input:
    x            variable (NC...)  Input data.
    is_training  bool              Whether to drop.
    rate         0. < float < 1.   Fraction dropped.

Output:
    y            variable (NC...)  Output data.
"""
spatial_dropout = Z.get('spatial_dropout', _my_spatial_dropout)


"""
1D spatial dropout (sequences).

Input:
    x            variable (NCW)    Input data.
    is_training  bool              Whether to drop.
    rate         0. < float < 1.   Fraction dropped.

Output:
    y            variable (NCW)    Output data.
"""
spatial_dropout1d = Z.get('spatial_dropout1d', _my_spatial_dropout)


"""
2D spatial dropout (images).

Input:
    x            variable (NCHW)   Input data.
    is_training  bool              Whether to drop.
    rate         0. < float < 1.   Fraction dropped.

Output:
    y            variable (NCHW)   Output data.
"""
spatial_dropout2d = Z.get('spatial_dropout2d', _my_spatial_dropout)


"""
3D spatial dropout (video).

Input:
    x            variable (NCDHW)  Input data.
    is_training  bool              Whether to drop.
    rate         0. < float < 1.   Fraction dropped.

Output:
    y            variable (NCDHW)  Output data.
"""
spatial_dropout3d = Z.get('spatial_dropout3d', _my_spatial_dropout)
