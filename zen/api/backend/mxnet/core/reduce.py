import mxnet as mx


def argmax(x, axis=-1):
    return mx.nd.argmax(x, axis)


def mean(x, axis=None, keepdims=False):
    if axis is None:
        axis = mx.base._Null
    ret = mx.nd.mean(x, axis, keepdims)
    if axis is mx.base._Null:
        ret = ret.asnumpy()[0]
    return ret
