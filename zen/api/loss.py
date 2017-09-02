from . import core as C


def binary_cross_entropy(true, pred):
    pred = C.clip(pred, C.epsilon(), 1. - C.epsilon())
    return -true * C.log(pred) - (1. - true) * C.log(1. - pred)


def categorical_cross_entropy(true, pred):
    pred = C.clip(pred, C.epsilon(), 1. - C.epsilon())
    ret = -true * C.log(pred)
    return C.mean(ret, -1)


def mean_squared_error(true, pred):
    ret = C.square(true - pred)
    return C.mean(ret, -1)
