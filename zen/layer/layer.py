from .. import variable


class Layerlike(object):
    def get_params(self):
        raise NotImplementedError

    def forward(self, x, is_training):
        raise NotImplementedError


class Layer(Layerlike):
    def __init__(self):
        super().__init__()
        self._params = []

    def add_param(self, arr):
        param = variable(arr)
        self._params.append(param)
        return param

    def get_params(self):
        return self._params


class Spec(object):
    def build(self, in_shape=None, in_dtype=None):
        raise NotImplementedError
