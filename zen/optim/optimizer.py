from copy import deepcopy


class Optimizee(object):
    def __init__(self, k2v):
        self.param = None
        self.__dict__.update(deepcopy(k2v))


class Optimizer(object):
    def __init__(self, defaults):
        self.defaults = defaults
        self.items = None

    def set_params(self, params):
        self.items = []
        for param in params:
            item = Optimizee(self.defaults)
            item.param = param
            self.items.append(item)

    def step(self):
        raise NotImplementedError
