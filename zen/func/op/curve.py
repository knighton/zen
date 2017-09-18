import numpy as np

from .. import backend as Z
from .. import core as C


def _elu(x, alpha=1.):
    """
    Exponential linear unit (ELU).
    """
    return C.maximum(0, x) + C.minimum(0, alpha * (C.exp(x) - 1.))


def _hard_shrink(x, lambda_=0.5):
    """
    Hard shrink.
    """
    return C.clip(x + lambda_, -np.inf, 0.) + C.clip(x - 0.5, 0., np.inf)


def _hard_sigmoid(x):
    """
    Hard sigmoid.
    """
    x = (0.2 * x) + 0.5
    return C.clip(x, 0, 1)


def _hard_tanh(x):
    """
    Hard tanh.
    """
    return C.clip(x, -1, 1)


def _leaky_relu(x, alpha=0.1):
    """
    Leaky rectified linear unit.
    """
    x = relu(x)
    if alpha != 0.:
        x -= alpha * relu(-x)
    return x


def _linear(x):
    """
    Linear activation (identity).
    """
    return x


def _relu(x, low=0., high=np.inf):
    """
    Rectified linear unit (ReLU).
    """
    return C.clip(x, low, high)


def _selu(x):
    """
    Scaled exponential linear unit (SELU).
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha)


def _sigmoid(x):
    """
    Sigmoid.
    """
    e_x = C.exp(x)
    return e_x / (e_x + 1.)


def _softmax(x):
    """
    Softmax (2D).
    """
    e_x = C.exp(x)
    return e_x / C.sum(e_x, 1, True)


def _softplus(x):
    """
    Softplus.
    """
    return C.log(1. + C.exp(x))


def _softshrink(x, lambda_=0.5):
    """
    Softshrink.
    """
    return C.sign(x) * C.maximum(C.abs(x) - lambda_, 0)


def _softsign(x):
    """
    Softsign.
    """
    return x / (1. + C.abs(x))


def _tanh_shrink(x):
    """
    TanH shrink.
    """
    return x - C.tanh(x)


"""
Activation functions.

    elu           (x, alpha=1.)
    hard_shrink   (x, lambda_=0.5)
    hard_sigmoid  (x)
    hard_tanh     (x)
    leaky_relu    (x, alpha=0.1)
    linear        (x)
    relu          (x, low=0., high=np.inf)
    selu          (x)
    sigmoid       (x)
    softmax       (x)
    softplus      (x)
    softshrink    (x, lambda_=0.5)
    softsign      (x)
    tanh          (x)
    tanh_shrink   (x)
"""
elu = Z.get('elu', _elu)
hard_shrink = Z.get('hard_shrink', _hard_shrink)
hard_sigmoid = Z.get('hard_sigmoid', _hard_sigmoid)
hard_tanh = Z.get('hard_tanh', _hard_tanh)
leaky_relu = Z.get('leaky_relu', _leaky_relu)
linear = _linear
relu = _relu
selu = Z.get('selu', _selu)
sigmoid = Z.get('sigmoid', _sigmoid)
softmax = Z.get('softmax', _softmax)
softplus = Z.get('softplus', _softplus)
softshrink = Z.get('softshrink', _softshrink)
softsign = Z.get('softsign', _softsign)
tanh = C.tanh
tanh_shrink = Z.get('tanh_shrink', _tanh_shrink)
