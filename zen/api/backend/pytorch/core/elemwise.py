import numpy as np


def abs(x):
    return x.abs()


def clip(x, min_value=None, max_value=None):
    if min_value is None:
        min_value = -np.inf
    if max_value is None:
        max_value = np.inf
    return x.clamp(min_value, max_value)


def neg(x):
    return x.neg()


def sign(x):
    return x.sign()


def exp(x):
    return x.exp()


def expm1(x):
    return x.expm1()


def log(x):
    return x.log()


def log1p(x):
    return x.log1p()


def pow(x, a):
    return x.pow(a)


def sqrt(x):
    return x.sqrt()


def rsqrt(x):
    return x.rsqrt()


def square(x):
    return x.square()
