import mxnet as mx


def argmax(x, axis=-1):
    return mx.nd.argmax(x, axis)


def argmin(x, axis=-1):
    return mx.nd.argmin(x, axis)


def _reduce(x, axis, keepdims, func_name):
    mx_axis = mx.base._Null if axis is None else axis
    func = getattr(mx.nd, func_name)
    return func(x, mx_axis, keepdims)


def max(x, axis=None, keepdims=False):
    return _reduce(x, axis, keepdims, 'max')


def mean(x, axis=None, keepdims=False):
    return _reduce(x, axis, keepdims, 'mean')


def min(x, axis=None, keepdims=False):
    return _reduce(x, axis, keepdims, 'min')


def prod(x, axis=None, keepdims=False):
    return _reduce(x, axis, keepdims, 'prod')


def sum(x, axis=None, keepdims=False):
    return _reduce(x, axis, keepdims, 'sum')
