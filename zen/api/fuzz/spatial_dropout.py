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


spatial_dropout = Z.get('spatial_dropout', _my_spatial_dropout)
spatial_dropout1d = Z.get('spatial_dropout1d', _my_spatial_dropout)
spatial_dropout2d = Z.get('spatial_dropout2d', _my_spatial_dropout)
spatial_dropout3d = Z.get('spatial_dropout3d', _my_spatial_dropout)
