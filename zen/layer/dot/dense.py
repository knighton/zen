from ... import init
from ..layer import Layer, Spec, Sugar


class DenseLayer(Layer):
    def __init__(self, w, b):
        super().__init__()
        self.w = self.add_param(w)
        self.b = self.add_param(b)

    def forward(self, x, is_training):
        return x.mm(self.w) + self.b


class DenseSpec(Spec):
    def __init__(self, out_dim=None):
        super().__init__()
        self.out_dim = out_dim

    def build(self, in_shape=None, in_dtype=None):
        in_dim, = in_shape
        if self.out_dim is None:
            out_dim = in_dim
        else:
            out_dim = self.out_dim
        out_shape = out_dim,
        layer = DenseLayer(init(in_dim, out_dim), init(out_dim))
        return layer, out_shape, in_dtype


Dense = Sugar(DenseSpec)
