import numpy as np


def abs(x):
    return x.abs()


def clip(x, low=-np.inf, high=np.inf):
    return x.clamp(low, high)


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


def sign(x):
    return x.sign()


def sqrt(x):
    return x.sqrt()


def square(x):
    return x.pow(2)
