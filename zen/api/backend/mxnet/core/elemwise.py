import mxnet as mx
import numpy as np


# Arithmetic.


def abs(x):
    return mx.nd.abs(x)


def clip(x, low=-np.inf, high=np.inf):
    return mx.nd.clip(x, low, high)


def neg(x):
    return mx.nd.negative(x)


def sign(x):
    return mx.nd.sign(x).astype('uint8')


# Exponentiation.


def exp(x):
    return mx.nd.exp(x)


def expm1(x):
    return mx.nd.expm1(x)


# Logarithms.


def log(x):
    return mx.nd.log(x)


def log10(x):
    return mx.nd.log10(x)


def log2(x):
    return mx.nd.log2(x)


def log1p(x):
    return mx.nd.log1p(x)


# Power.


def pow(x, a):
    return mx.nd.power(x, a)


def sqrt(x):
    return mx.nd.sqrt(x)


def rsqrt(x):
    return mx.nd.rsqrt(x)


def square(x):
    return mx.nd.square(x)


# Rounding.


def ceil(x):
    return mx.nd.ceil(x)


def floor(x):
    return mx.nd.floor(x)


def round(x):
    return mx.nd.round(x)


def trunc(x):
    return mx.nd.trunc(x)


# Trigonometry.


def sin(x):
    return mx.nd.sin(x)


def cos(x):
    return mx.nd.cos(x)


def tan(x):
    return mx.nd.tan(x)


def arcsin(x):
    return mx.nd.arcsin(x)


def arccos(x):
    return mx.nd.arccos(x)


def arctan(x):
    return mx.nd.arctan(x)


def sinh(x):
    return mx.nd.sinh(x)


def cosh(x):
    return mx.nd.cosh(x)


def tanh(x):
    return mx.nd.tanh(x)


def arcsinh(x):
    return mx.nd.arcsinh(x)


def arccosh(x):
    return mx.nd.arccosh(x)


def arctanh(x):
    return mx.nd.arctanh(x)
