import mxnet as mx

from ... import base as B
from ... import core as C


def avg_pool(x, window, padding, stride):
    ndim = C.ndim(x)
    window = B.to_shape(window, ndim)
    padding = B.to_shape(padding, ndim)
    stride = B.to_shape(padding, ndim)
    return mx.nd.Pooling(data=x, kernel=window, pad=padding, stride=stride,
                         pool_type='avg')


avg_pool1d = avg_pool
avg_pool2d = avg_pool
avg_pool3d = avg_pool


def max_pool(x, window, padding, stride):
    ndim = C.ndim(x) - 2
    window = B.to_shape(window, ndim)
    padding = B.to_shape(padding, ndim)
    stride = B.to_shape(stride, ndim)
    return mx.nd.Pooling(data=x, kernel=window, pad=padding, stride=stride,
                         pool_type='max')


max_pool1d = max_pool
max_pool2d = max_pool
max_pool3d = max_pool
