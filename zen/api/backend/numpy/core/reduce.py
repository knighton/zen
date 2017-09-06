def all(x, axis=None, keepdims=False):
    return x.all(axis, 'uint8', keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    return x.any(axis, 'uint8', keepdims=keepdims)


def argmax(x, axis=-1):
    return x.argmax(axis)


def argmin(x, axis=-1):
    return x.argmin(axis)


def max(x, axis=None, keepdims=False):
    return x.max(axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    return x.mean(axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    return x.min(axis, keepdims=keepdims)


def prod(x, axis=None, keepdims=False):
    return x.prod(axis, keepdims=keepdims)


def std(x, axis=None, keepdims=False):
    return x.std(axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    return x.sum(axis, keepdims=keepdims)


def var(x, axis=None, keepdims=False):
    return x.var(axis, keepdims=keepdims)
