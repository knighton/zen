import mxnet as mx
import numpy as np


def clip(x, min_value=-np.inf, max_value=np.inf):
    return mx.nd.clip(x, min_value, max_value)


def log(x):
    return mx.nd.log(x)


def round(x):
    return mx.nd.round(x)


def sqrt(x):
    return mx.nd.sqrt(x)


def square(x):
    return mx.nd.square(x)
