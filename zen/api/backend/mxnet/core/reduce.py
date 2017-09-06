import mxnet as mx


def argmax(x, axis=-1):
    return mx.nd.argmax(x, axis)


def argmin(x, axis=-1):
    return mx.nd.argmin(x, axis)


def _mx_axis(axis):
    return mx.base._Null if axis is None else axis


def max(x, axis=None, keepdims=False):
    return mx.nd.max(x, _mx_axis(axis), keepdims)


def mean(x, axis=None, keepdims=False):
    return mx.nd.mean(x, _mx_axis(axis), keepdims)


def min(x, axis=None, keepdims=False):
    return mx.nd.max(x, _mx_axis(axis), keepdims)


def prod(x, axis=None, keepdims=False):
    return mx.nd.prod(x, _mx_axis(axis), keepdims)


def sum(x, axis=None, keepdims=False):
    return mx.nd.sum(x, _mx_axis(axis), keepdims)
