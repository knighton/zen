import mxnet as mx

from .....core.util import to_shape
from ... import core as C


def avg_pool(x, window, padding, stride):
    ndim = C.get_ndim(x)
    window = to_shape(window, ndim)
    padding = to_shape(padding, ndim)
    stride = to_shape(padding, ndim)
    return mx.nd.Pooling(data=x, kernel=window, pad=padding, stride=stride,
                         pool_type='avg')


avg_pool1d = avg_pool
avg_pool2d = avg_pool
avg_pool3d = avg_pool


def max_pool(x, window, padding, stride):
    ndim = C.get_ndim(x) - 2
    window = to_shape(window, ndim)
    padding = to_shape(padding, ndim)
    stride = to_shape(stride, ndim)
    return mx.nd.Pooling(data=x, kernel=window, pad=padding, stride=stride,
                         pool_type='max')


max_pool1d = max_pool
max_pool2d = max_pool
max_pool3d = max_pool
