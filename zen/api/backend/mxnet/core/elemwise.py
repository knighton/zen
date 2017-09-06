import mxnet as mx
import numpy as np


def abs(x):
    return mx.nd.abs(x)


def clip(x, low=-np.inf, high=np.inf):
    return mx.nd.clip(x, low, high)


def neg(x):
    return mx.nd.negative(x)


def sign(x):
    return mx.nd.sign(x).astype('uint8')


def exp(x):
    return mx.nd.exp(x)


def expm1(x):
    return mx.nd.expm1(x)


def log(x):
    return mx.nd.log(x)


def log10(x):
    return mx.nd.log10(x)


def log2(x):
    return mx.nd.log2(x)


def log1p(x):
    return mx.nd.log1p(x)


def pow(x, a):
    return mx.nd.power(x, a)


def sqrt(x):
    return mx.nd.sqrt(x)


def rsqrt(x):
    return mx.nd.rsqrt(x)


def square(x):
    return mx.nd.square(x)
