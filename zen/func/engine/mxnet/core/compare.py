import mxnet as mx


def equal(x, y):
    return mx.nd.broadcast_equal(x, y)


def greater_equal(x, y):
    return mx.nd.broadcast_greater_equal(x, y)


def greater(x, y):
    return mx.nd.broadcast_greater(x, y)


def less_equal(x, y):
    return mx.nd.broadcast_less_equal(x, y)


def less(x, y):
    return mx.nd.broadcast_less(x, y)


def maximum(x, y):
    return mx.nd.broadcast_maximum(x, y)


def minimum(x, y):
    return mx.nd.broadcast_minimum(x, y)


def not_equal(x, y):
    return mx.nd.broadcast_not_equal(x, y)
