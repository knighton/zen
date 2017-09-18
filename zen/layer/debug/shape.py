from ... import func as Z
from ..base import Transform, TransformSpec, Sugar


def _err(got_shape, want_shape, name):
    if name:
        ret = 'Shape of %s is wrong: expected %s, but got %s.' % \
            (name, want_shape, got_shape)
    else:
        ret = 'Shape is wrong: expected %s, but got %s.' % \
            (want_shape, got_shape)
    return ret


class ShapeLayer(Transform):
    def __init__(self, shape, name):
        super().__init__()
        self.shape = shape
        self.name = name

    def forward(self, x, is_training):
        x_shape = Z.shape(x)[1:]
        assert x_shape == self.shape, _err(x_shape, self.shape, self.name)
        return x


class ShapeSpec(TransformSpec):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def build(self, in_shape, in_dtype):
        if self.name:
            print('Shape of %s: %s' % (self.name, in_shape))
        else:
            print('Shape: %s' % (in_shape,))
        return ShapeLayer(in_shape, self.name), in_shape, in_dtype


Shape = Sugar(ShapeSpec)
