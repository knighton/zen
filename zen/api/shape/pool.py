from torch.nn import functional as F

from ..core.data import ndim
from ..core.util import to_one
from .pad import pad3d


def avg_pool1d(x, window, padding, stride):
    """
    x        tensor (batch_size, channels, in_width)
    window   dim or (width,)
    padding  dim or (width,)
    stride   dim or (width,)
    """
    window = to_one(window)
    padding = to_one(padding)
    stride = to_one(stride)
    return F.avg_pool1d(x, window, stride, padding)


def avg_pool2d(x, window, padding, stride):
    """
    x        tensor (batch_size, channels, in_height, in_width)
    window   dim or (height, width)
    padding  dim or (height, width)
    stride   dim or (height, width)
    """
    return F.avg_pool2d(x, window, stride, padding)


def avg_pool3d(x, window, padding, stride):
    """
    x        tensor (batch_size, channels, in_depth, in_height, in_width)
    window   dim or (depth, height, width)
    padding  dim or (depth, height, width)
    stride   dim or (depth, height, width)
    """
    x = pad3d(x, padding)
    return F.avg_pool3d(x, window, stride)


_DIM2AVG_POOL = {
    1: avg_pool1d,
    2: avg_pool2d,
    3: avg_pool3d,
}


def avg_pool(x, window, padding, stride):
    """
    x        tensor (batch_size, channels, in_shape...)
    window   dim or shape
    padding  dim or shape
    stride   dim or shape
    """
    dim = ndim(x) - 2
    return _DIM2AVG_POOL[dim](x, window, padding, stride)


def max_pool1d(x, window, padding, stride):
    """
    x        tensor (batch_size, channels, in_width)
    window   dim or (width,)
    padding  dim or (width,)
    stride   dim or (width,)
    """
    window = to_one(window)
    padding = to_one(padding)
    stride = to_one(stride)
    return F.max_pool1d(window, stride, padding)


def max_pool2d(x, window, padding, stride):
    """
    x        tensor (batch_size, channels, in_height, in_width)
    window   dim or (height, width)
    padding  dim or (height, width)
    stride   dim or (height, width)
    """
    return F.max_pool2d(window, stride, padding)


def max_pool3d(x, window, padding, stride):
    """
    x        tensor (batch_size, channels, in_depth, in_height, in_width)
    window   dim or (depth, height, width)
    padding  dim or (depth, height, width)
    stride   dim or (depth, height, width)
    """
    return F.max_pool3d(window, stride, padding)


_DIM2MAX_POOL = {
    1: max_pool1d,
    2: max_pool2d,
    3: max_pool3d,
}


def max_pool(x, window, padding, stride):
    """
    x        tensor (batch_size, channels, in_shape...)
    window   dim or shape
    padding  dim or shape
    stride   dim or shape
    """
    dim = ndim(x) - 2
    return _DIM2MAX_POOL[dim](x, window, padding, stride)
