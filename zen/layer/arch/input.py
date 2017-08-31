from ..layer import Layer, Spec


class InputLayer(Layer):
    def forward(self, x, is_training):
        return x


class InputSpec(Spec):
    def __init__(self, shape, dtype):
        super().__init__()
        self.shape = shape
        self.dtype = dtype

    def build(self, in_shape=None, in_dtype=None):
        if in_shape is not None:
            assert in_shape == self.shape
        if in_dtype is not None:
            assert in_dtype == self.dtype
        return InputLayer(), self.shape, self.dtype
