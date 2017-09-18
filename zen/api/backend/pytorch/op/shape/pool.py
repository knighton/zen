from torch.nn import functional as F

from .....core.util import to_one
from ... import core as C
from .pad import constant_pad3d


def avg_pool1d(x, window, padding, stride):
    window = to_one(window)
    padding = to_one(padding)
    stride = to_one(stride)
    return F.avg_pool1d(x, window, stride, padding)


def avg_pool2d(x, window, padding, stride):
    return F.avg_pool2d(x, window, stride, padding)


def avg_pool3d(x, window, padding, stride):
    x = constant_pad3d(x, padding, 0)
    return F.avg_pool3d(x, window, stride)


_DIM2AVG_POOL = {
    1: avg_pool1d,
    2: avg_pool2d,
    3: avg_pool3d,
}


def avg_pool(x, window, padding, stride):
    dim = C.get_ndim(x) - 2
    return _DIM2AVG_POOL[dim](x, window, padding, stride)


def max_pool1d(x, window, padding, stride):
    window = to_one(window)
    padding = to_one(padding)
    stride = to_one(stride)
    return F.max_pool1d(x, window, stride, padding)


def max_pool2d(x, window, padding, stride):
    return F.max_pool2d(x, window, stride, padding)


def max_pool3d(x, window, padding, stride):
    return F.max_pool3d(x, window, stride, padding)


_DIM2MAX_POOL = {
    1: max_pool1d,
    2: max_pool2d,
    3: max_pool3d,
}


def max_pool(x, window, padding, stride):
    dim = C.get_ndim(x) - 2
    return _DIM2MAX_POOL[dim](x, window, padding, stride)
