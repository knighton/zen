import mxnet as mx
import numpy as np


def abs(x):
    return mx.nd.abs(x)


def clip(x, low=-np.inf, high=np.inf):
    return mx.nd.clip(x, low, high)


def exp(x):
    return mx.nd.exp(x)


def log(x):
    return mx.nd.log(x)


def pow(x, a):
    return mx.nd.power(x, a)


def round(x):
    return mx.nd.round(x)


def sign(x):
    return mx.nd.sign(x).astype('uint8')


def sqrt(x):
    return mx.nd.sqrt(x)


def square(x):
    return mx.nd.square(x)
