import sys

from . import func as Z


class Optimizer(object):
    def __init__(self, item_init, item_step, config):
        self.item_init = item_init
        self.item_step = item_step
        self.config = config
        self.items = None

    def set_params(self, params):
        self.items = []
        for param in params:
            item = self.item_init(param, **self.config)
            self.items.append(item)

    def step(self):
        for item in self.items:
            self.item_step(item)


def sgd(**config):
    return Optimizer(Z.sgd_init, Z.sgd_step, config)


def sgd_momentum(**config):
    return Optimizer(Z.sgd_momentum_init, Z.sgd_momentum_step, config)


sgdm = sgd_momentum


def rmsprop(**config):
    return Optimizer(Z.rmsprop_init, Z.rmsprop_step, config)


def adam(**config):
    return Optimizer(Z.adam_init, Z.adam_step, config)


def get(x):
    if isinstance(x, Optimizer):
        return x
    elif isinstance(x, str):
        module = sys.modules[__name__]
        return getattr(module, x)()
    else:
        assert False
