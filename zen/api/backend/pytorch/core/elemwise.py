import numpy as np


# Arithmetic.


def abs(x):
    return x.abs()


def clip(x, low=-np.inf, high=np.inf):
    return x.clamp(low, high)


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


# Power.


def pow(x, a):
    return x.pow(a)


def sqrt(x):
    return x.sqrt()


def rsqrt(x):
    return x.rsqrt()


def square(x):
    return x.pow(2)


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
