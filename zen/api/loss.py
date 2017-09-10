from . import core as C


def binary_cross_entropy(true, pred):
    if C.get_ndim(true) == 1:
        true = C.expand_dims(true, 1)
    pred = C.clip(pred, C.epsilon(), 1. - C.epsilon())
    return -true * C.log(pred) - (1. - true) * C.log(1. - pred)


def categorical_cross_entropy(true, pred):
    pred = C.clip(pred, C.epsilon(), 1. - C.epsilon())
    ret = -true * C.log(pred)
    return C.mean(ret, -1)


def categorical_hinge(true, pred):
    pos = C.sum(true * pred, -1)
    neg = C.max((1. - true) * pred, -1)
    return C.maximum(0., neg - pos + 1.)


def cosine_proximity(true, pred):
    true = C.l2_normalize(true, -1)
    pred = C.l2_normalize(pred, -1)
    ret = -true * pred
    return C.mean(ret, -1)


def hinge(true, pred):
    ret = C.maximum(1. - true * pred, 0.)
    return C.mean(ret, -1)


def kullback_leibler_divergence(true, pred):
    true = C.clip(true, C.epsilon(), 1.)
    pred = C.clip(pred, C.epsilon(), 1.)
    ret = true * C.log(true / pred)
    return C.mean(ret, -1)


def logcosh(true, pred):
    x = pred - true
    ret = C.log(C.cosh(x))
    return C.mean(ret, -1)


def mean_absolute_error(true, pred):
    ret = C.abs(pred - true)
    return C.mean(ret, -1)


def mean_absolute_percentage_error(true, pred):
    num = pred - true
    denom = C.clip(C.abs(true), C.epsilon())
    diff = C.abs(num / denom)
    return 100. * C.mean(diff, -1)


def mean_squared_error(true, pred):
    ret = C.square(pred - true)
    return C.mean(ret, -1)


def mean_squared_logarithmic_error(true, pred):
    pred = C.clip(pred, C.epsilon())
    pred = C.log1p(pred)
    true = C.clip(true, C.epsilon())
    true = C.log1p(true)
    ret = C.square(pred - true)
    return C.mean(ret, -1)


def poisson(true, pred):
    ret = pred - true * C.log(pred + C.epsilon())
    return C.mean(ret, -1)


def squared_hinge(true, pred):
    ret = C.square(C.maximum(1. - true * pred, 0.))
    return C.mean(ret, -1)


mse = mean_squared_error
mae = mean_absolute_error
mape = mean_absolute_percentage_error
msle = mean_squared_logarithmic_error
kld = kullback_leibler_divergence
cosine = cosine_proximity
