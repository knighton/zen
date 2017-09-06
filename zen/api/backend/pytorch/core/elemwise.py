import numpy as np


def abs(x):
    return x.abs()


def clip(x, low=-np.inf, high=np.inf):
    return x.clamp(low, high)


def cos(x):
    return x.cos()


def cumprod(x, axis=0):
    return x.cumprod(axis)


def cumsum(x, axis=0):
    return x.cumsum(axis)


def exp(x):
    return x.exp()


def log(x):
    return x.log()


def pow(x, a):
    return x.pow(a)


def round(x):
    return x.round()


def sign(x):
    return x.sign()


def sin(x):
    return x.sin()


def sqrt(x):
    return x.sqrt()


def square(x):
    return x.pow(2)
