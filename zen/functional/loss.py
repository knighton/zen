from .core.epsilon import epsilon


def binary_cross_entropy(true, pred):
    pred = pred.clamp(epsilon(), 1. - epsilon())
    return -true * pred.log() - (1. - true) * (1. - pred).log()


def categorical_cross_entropy(true, pred):
    pred = pred.clamp(epsilon(), 1. - epsilon())
    ret = -true * pred.log()
    return ret.mean()


def mean_squared_error(true, pred):
    return (true - pred).pow(2).mean()
