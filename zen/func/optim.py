import numpy as np

from . import base as B
from . import core as C


class Optimizee(object):
    def __init__(self, var, **kwargs):
        self.variable = var
        self.data = C.data(var)
        assert 'variable' not in kwargs
        assert 'data' not in kwargs
        assert 'grad' not in kwargs
        self.__dict__.update(kwargs)

    def grad(self):
        return C.gradient(self.variable)

    def update(self, delta):
        C.update(self.variable, self.data + delta)


def sgd_init(var, lr=0.01):
    assert 0. < lr
    return Optimizee(var, **{'lr': lr})


def sgd_step(x):
    x.update(-x.lr * x.grad())


def zeros_like(var):
    return C.tensor(np.zeros(C.get_shape(var), B.floatx()))


def sgd_momentum_init(var, lr=0.01, momentum=0.9):
    assert 0. < lr
    assert 0. <= momentum <= 1.
    return Optimizee(var, **{
        'lr': lr,
        'momentum': momentum,
        'velocity': zeros_like(var),
    })


def sgd_momentum_step(x):
    x.velocity = x.velocity * x.momentum - x.lr * x.grad()
    x.update(x.velocity)


def rmsprop_init(var, decay_rate=0.99, epsilon=1e-6, lr=0.01):
    assert 0. < decay_rate < 1.
    assert 0. < epsilon
    assert 0. < lr
    return Optimizee(var, **{
        'cache': zeros_like(var),
        'decay_rate': decay_rate,
        'epsilon': epsilon,
        'lr': lr,
    })


def rmsprop_step(x):
    grad = x.grad()
    x.cache = x.decay_rate * x.cache + (1. - x.decay_rate) * C.square(grad)
    x.update(-x.lr * grad / (C.sqrt(x.cache) + x.epsilon))


def adam_init(var, beta1=0.9, beta2=0.99, epsilon=1e-6, lr=1e-3):
    assert 0. < beta1 < 1.
    assert 0. < beta2 < 1.
    assert 0. < epsilon
    assert 0. < lr
    return Optimizee(var, **{
        'beta1': beta1,
        'beta2': beta2,
        'epsilon': epsilon,
        'lr': lr,
        'm': zeros_like(var),
        'v': zeros_like(var),
        't': 1,
    })


def adam_step(x):
    grad = x.grad()
    x.m = x.beta1 * x.m + (1. - x.beta1) * grad
    x.v = x.beta2 * x.v + (1. - x.beta2) * C.square(grad)
    scaled_m = x.m / (1. - x.beta1 ** x.t)
    scaled_v = x.v / (1. - x.beta2 ** x.t)
    x.update(-x.lr * scaled_m / (C.sqrt(scaled_v) + x.epsilon))
