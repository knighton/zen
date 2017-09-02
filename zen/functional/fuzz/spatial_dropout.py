import numpy as np
from torch.nn import functional as F

from ..core.data import constant, shape


def _my_spatial_dropout(x, is_training, rate):
    if not is_training:
        return x
    shape = shape(x)
    noise_shape = shape[:2] + (1,) * (len(shape) - 2)
    max_value = 1. / rate
    mask = np.random.uniform(0, max_value, noise_shape).astype('float32')
    mask = np.floor(mask.clip(0., 1.))
    mask = constant(mask)
    return x * mask / (1. - rate)


spatial_dropout1d = _my_spatial_dropout


def spatial_dropout2d(x, is_training, rate):
    return F.dropout2d(x, rate, is_training, False)


def spatial_dropout3d(x, is_training, rate):
    return F.dropout3d(x, rate, is_training, False)


_DIM2SPATIAL_DROPOUT = {
    1: spatial_dropout1d,
    2: spatial_dropout2d,
    2: spatial_dropout3d,
}


def spatial_dropout(x, is_training, rate):
    dim = x.size() - 2
    return _DIM2SPATIAL_DROPOUT[dim](x, is_training, rate)
