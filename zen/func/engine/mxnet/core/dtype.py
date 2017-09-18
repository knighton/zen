from .. import base as B


def cast(x, dtype=None):
    dtype = dtype or B.floatx()
    return x.astype(dtype)


def get_dtype(x):
    return x.dtype.__name__
