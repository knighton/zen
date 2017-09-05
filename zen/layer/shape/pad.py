from ... import api as Z
from ..layer import Layer, Spec, Sugar


class PadLayer(Layer):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding


class PadSpec(Spec):
    def __init__(self, padding, ndim=None):
        """
        padding  {dim, shape, pairs}  Int means repeat per dimension.
        ndim     {None, 1, 2, 3}      Specifies dimensionality of input.
        """
        super().__init__()
        Z.check_input_ndim(ndim, {1, 2, 3}, 'ndim')
        self.padding = Z.normalize_int_padding(padding, ndim, 'padding')
        self.ndim = ndim

    def make_layer(self):
        raise NotImplementedError

    def build(self, in_shape, in_dtype):
        ndim = Z.verify_input_ndim(self.ndim, in_shape)
        out_shape = Z.pad_out_shape(in_shape, self.padding)
        return self.make_layer(), out_shape, in_dtype


class ConstantPadLayer(PadLayer):
    def __init__(self, padding, value):
        self.padding = padding
        self.value = value

    def forward(self, x, is_training):
        return Z.constant_pad(x, self.padding, self.value)


class ConstantPadSpec(PadSpec):
    def __init__(self, padding, value, ndim=None):
        """
        padding  {dim, shape, pairs}  Int means repeat per dimension.
        value    {int, float}         Pad value (eg, zero).
        ndim     {None, 1, 2, 3}      Specifies dimensionality of input.
        """
        super().__init__(padding, ndim)
        self.value = value

    def make_layer(self):
        return ConstantPadLayer(self.padding, self.value)


ConstantPad = Sugar(ConstantPadSpec)
ConstantPad1D = Sugar(ConstantPadSpec, {'ndim': 1})
ConstantPad2D = Sugar(ConstantPadSpec, {'ndim': 2})
ConstantPad3D = Sugar(ConstantPadSpec, {'ndim': 3})

ZeroPad = Sugar(ConstantPadSpec, {'value': 0})
ZeroPad1D = Sugar(ConstantPadSpec, {'value': 0, 'ndim': 1})
ZeroPad2D = Sugar(ConstantPadSpec, {'value': 0, 'ndim': 2})
ZeroPad3D = Sugar(ConstantPadSpec, {'value': 0, 'ndim': 3})
