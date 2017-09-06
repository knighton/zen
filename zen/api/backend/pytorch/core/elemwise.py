import numpy as np


# Arithmetic.


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


# Exponentiation.


def exp(x):
    return x.exp()


def expm1(x):
    return x.expm1()


# Logarithms.


def log(x):
    return x.log()


def log1p(x):
    return x.log1p()


def pow(x, a):
    return x.pow(a)


# Power.


def sqrt(x):
    return x.sqrt()


def rsqrt(x):
    return x.rsqrt()


def square(x):
    return x.square()


# Rounding.


def ceil(x):
    return x.ceil()


def floor(x):
    return x.floor()


def round(x):
    return x.round()


def trunc(x):
    return x.trunc()


# Trigonometry.


def sin(x):
    return x.sin()


def cos(x):
    return x.cos()


def tan(x):
    return x.tan()


def arcsin(x):
    return x.asin()


def arccos(x):
    return x.acos()


def arctan(x):
    return x.atan()


def sinh(x):
    return x.sinh()


def cosh(x):
    return x.cosh()


def tanh(x):
    return x.tanh()
