import numpy as np


# Arithmetic.


def abs(x):
    return np.abs(x)


def clip(x, low=-np.inf, high=np.inf):
    return x.clip(low, high)


def neg(x):
    return np.negative(x)


def sign(x):
    return np.sign(x)


# Exponentiation.


def exp(x):
    return np.exp(x)


def expm1(x):
    return np.expm1(x)


# Logarithms.


def log(x):
    return np.log(x)


def log1p(x):
    return np.log1p(x)


# Power.


def pow(x, a):
    return np.power(x, a)


def sqrt(x):
    return np.sqrt(x)


def rsqrt(x):
    return 1. / np.sqrt(x)


def square(x):
    return np.square(x)


# Rounding.


def ceil(x):
    return np.ceil(x)


def floor(x):
    return np.floor(x)


def round(x):
    return np.round(x)


def trunc(x):
    return np.trunc(x)


# Trigonometry.


def sin(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


def tan(x):
    return np.tan(x)


def arcsin(x):
    return np.arcsin(x)


def arccos(x):
    return np.arccos(x)


def arctan(x):
    return np.arctan(x)


def sinh(x):
    return np.sinh(x)


def cosh(x):
    return np.cosh(x)


def tanh(x):
    return x.tanh(x)


def arcsinh(x):
    return np.arcsinh(x)


def arccosh(x):
    return np.arccosh(x)


def arctanh(x):
    return np.arctanh(x)
