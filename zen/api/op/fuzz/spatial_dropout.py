import numpy as np

from ... import backend as Z
from ... import core as C


def _spatial_dropout(x, is_training, rate):
    if not is_training:
        return x
    shape = C.get_shape(x)
    noise_shape = shape[:2] + (1,) * (len(shape) - 2)
    max_value = 1. / rate
    mask = np.random.uniform(0, max_value, noise_shape).astype('float32')
    mask = np.floor(mask.clip(0., 1.))
    mask = C.constant(mask)
    return x * mask / (1. - rate)


"""
Spatial ropout.

    x, is_training, rate -> y

Input:
                           0D  1D   2D    3D
                           --  --   --    --
    x            variable  NC  NCW  NCHW  NCDHW
    is_training  bool
    rate         0 to 1

Output:
    y            variable  NC  NCW  NCHW  NCDHW
"""
spatial_dropout = Z.get('spatial_dropout', _spatial_dropout)
spatial_dropout1d = Z.get('spatial_dropout1d', _spatial_dropout)
spatial_dropout2d = Z.get('spatial_dropout2d', _spatial_dropout)
spatial_dropout3d = Z.get('spatial_dropout3d', _spatial_dropout)
