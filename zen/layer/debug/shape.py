from ... import func as Z
from ... import init
from ..base import Transform, TransformSpec, Sugar


class ShapeLayer(Transform):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.has_printed = False

    def forward(self, x, is_training):
        if not self.has_printed:
            ss = []
            ss.append('[forward shape]')
            if self.name:
                ss.append('[%s]' % self.name)
            ss.append(str(Z.shape(x)))
            print(' '.join(ss))
            self.has_printed = True
        return x


class ShapeSpec(TransformSpec):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def build(self, in_shape, in_dtype):
        ss = []
        ss.append('[build shape]')
        if self.name:
            ss.append('[%s]' % self.name)
        ss.append(str(in_shape))
        print(' '.join(ss))
        return ShapeLayer(self.name), in_shape, in_dtype


Shape = Sugar(ShapeSpec)
