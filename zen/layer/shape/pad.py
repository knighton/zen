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
        return self.make_layer(ndim), out_shape, in_dtype


class ConstantPadLayer(PadLayer):
    def __init__(self, ndim, padding, value):
        super().__init__(padding)
        self.value = value
        self.constant_pad = Z.get('constant_pad', ndim)

    def forward(self, x, is_training):
        return self.constant_pad(x, self.padding, self.value)


class ConstantPadSpec(PadSpec):
    def __init__(self, padding, value, ndim=None):
        """
        padding  {dim, shape, pairs}  Int means repeat per dimension.
        value    {int, float}         Pad value (eg, zero).
        ndim     {None, 1, 2, 3}      Specifies dimensionality of input.
                                      None means derive from input in build().
        """
        super().__init__(padding, ndim)
        self.value = value

    def make_layer(self, ndim):
        return ConstantPadLayer(ndim, self.padding, self.value)


ConstantPad = Sugar(ConstantPadSpec)
ConstantPad1D = Sugar(ConstantPadSpec, {'ndim': 1})
ConstantPad2D = Sugar(ConstantPadSpec, {'ndim': 2})
ConstantPad3D = Sugar(ConstantPadSpec, {'ndim': 3})

ZeroPad = Sugar(ConstantPadSpec, {'value': 0})
ZeroPad1D = Sugar(ConstantPadSpec, {'value': 0, 'ndim': 1})
ZeroPad2D = Sugar(ConstantPadSpec, {'value': 0, 'ndim': 2})
ZeroPad3D = Sugar(ConstantPadSpec, {'value': 0, 'ndim': 3})


class EdgePadLayer(PadLayer):
    def __init__(self, ndim, padding):
        super().__init__(padding)
        self.edge_pad = Z.get('edge_pad', ndim)

    def forward(self, x, is_training):
        return self.edge_pad(x, self.padding)


class EdgePadSpec(PadSpec):
    def make_layer(self, ndim):
        return EdgePadLayer(ndim, self.padding)


EdgePad = Sugar(EdgePadSpec)
EdgePad1D = Sugar(EdgePadSpec, {'ndim': 1})
EdgePad2D = Sugar(EdgePadSpec, {'ndim': 2})
EdgePad3D = Sugar(EdgePadSpec, {'ndim': 3})


class ReflectPadLayer(PadLayer):
    def __init__(self, ndim, padding):
        super().__init__(padding)
        self.reflect_pad = Z.get('reflect_pad', ndim)

    def forward(self, x, is_training):
        return self.reflect_pad(x, self.padding)


class ReflectPadSpec(PadSpec):
    def make_layer(self, ndim):
        return ReflectPadLayer(ndim, self.padding)


ReflectPad = Sugar(ReflectPadSpec)
ReflectPad1D = Sugar(ReflectPadSpec, {'ndim': 1})
ReflectPad2D = Sugar(ReflectPadSpec, {'ndim': 2})
ReflectPad3D = Sugar(ReflectPadSpec, {'ndim': 3})
