from .floatx import floatx


def cast(x, dtype=None):
    dtype = dtype or floatx()
    return x.astype(dtype)


def get_dtype(x):
    return x.dtype.__name__
