import numpy as np

from . import backend as Z
from . import core as C


def _my_elu(x, alpha=1.):
    """
    Exponential linear unit (ELU).
    """
    return C.maximum(0, x) + C.minimum(0, alpha * (C.exp(x) - 1.))


def _my_hard_shrink(x, lambda_=0.5):
    """
    Hard shrink.
    """
    return C.clip(x + lambda_, -np.inf, 0.) + C.clip(x - 0.5, 0., np.inf)


def _my_hard_sigmoid(x):
    """
    Hard sigmoid.
    """
    x = (0.2 * x) + 0.5
    return C.clip(x, 0, 1)


def _my_hard_tanh(x):
    """
    Hard tanh.
    """
    return C.clip(x, -1, 1)


def _my_leaky_relu(x, alpha=0.1):
    """
    Leaky rectified linear unit.
    """
    x = relu(x)
    if alpha != 0.:
        x -= alpha * relu(-x)
    return x


def _my_linear(x):
    """
    Linear activation (identity).
    """
    return x


def _my_relu(x, low=0., high=np.inf):
    """
    Rectified linear unit (ReLU).
    """
    return C.clip(x, low, high)


def _my_selu(x):
    """
    Scaled exponential linear unit (SELU).
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha)


def _my_sigmoid(x):
    """
    Sigmoid.
    """
    e_x = C.exp(x)
    return e_x / (e_x + 1.)


def _my_softmax(x):
    """
    Softmax (2D).
    """
    e_x = C.exp(x)
    return e_x / C.sum(e_x, 1, True)


def _my_softplus(x):
    """
    Softplus.
    """
    return C.log(1. + C.exp(x))


def _my_softshrink(x, lambda_=0.5):
    """
    Softshrink.
    """
    return C.sign(x) * C.maximum(C.abs(x) - lambda_, 0)


def _my_softsign(x):
    """
    Softsign.
    """
    return x / (1. + C.abs(x))


def _my_tanh_shrink(x):
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
elu = Z.get('elu', _my_elu)
hard_shrink = Z.get('hard_shrink', _my_hard_shrink)
hard_sigmoid = Z.get('hard_sigmoid', _my_hard_sigmoid)
hard_tanh = Z.get('hard_tanh', _my_hard_tanh)
leaky_relu = Z.get('leaky_relu', _my_leaky_relu)
linear = _my_linear
relu = _my_relu
selu = Z.get('selu', _my_selu)
sigmoid = Z.get('sigmoid', _my_sigmoid)
softmax = Z.get('softmax', _my_softmax)
softplus = Z.get('softplus', _my_softplus)
softshrink = Z.get('softshrink', _my_softshrink)
softsign = Z.get('softsign', _my_softsign)
tanh = C.tanh
tanh_shrink = Z.get('tanh_shrink', _my_tanh_shrink)
