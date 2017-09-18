from .base.float import floatx


def cast(x, dtype=None):
    dtype = dtype or floatx()
    return x.astype(dtype)


def dtype(x):
    return x.dtype.__name__
