from .core.elemwise import clip, log, square
from .core.epsilon import epsilon
from .core.reduce import mean


def binary_cross_entropy(true, pred):
    pred = clip(pred, epsilon(), 1. - epsilon())
    return -true * log(pred) - (1. - true) * log(1. - pred)


def categorical_cross_entropy(true, pred):
    pred = clip(pred, epsilon(), 1. - epsilon())
    ret = -true * log(pred)
    return mean(ret, -1)


def mean_squared_error(true, pred):
    ret = square(true - pred)
    return mean(ret, -1)
