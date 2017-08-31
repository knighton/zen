from ..layer import Layerlike, Spec


class SequenceLayer(Layerlike):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params

    def forward(self, x, is_training):
        for layer in self.layers:
            x = layer.forward(x, is_training)
        return x


class SequenceSpec(Spec):
    def __init__(self, *specs):
        super().__init__()
        self.specs = specs

    def build(self, in_shape=None, in_dtype=None):
        layers = []
        for spec in self.specs:
            layer, in_shape, in_dtype = spec.build(in_shape, in_dtype)
            layers.append(layer)
        return SequenceLayer(layers), in_shape, in_dtype
