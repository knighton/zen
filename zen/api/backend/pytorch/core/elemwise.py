import numpy as np


def abs(x):
    return x.abs()


def clip(x, low=-np.inf, high=np.inf):
    return x.clamp(low, high)


def exp(x):
    return x.exp()


def log(x):
    return x.log()


def round(x):
    return x.round()


def sign(x):
    return x.sign()


def sqrt(x):
    return x.sqrt()


def square(x):
    return x.pow(2)
