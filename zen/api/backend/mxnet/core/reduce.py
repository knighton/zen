import mxnet as mx


def argmax(x, axis=-1):
    return mx.nd.argmax(x, axis)


def mean(x, axis=None, keepdims=False):
    if axis is None:
        axis = mx.base._Null
    return mx.nd.mean(x, axis, keepdims)
