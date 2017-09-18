from ... import func as Z
from ..base import Merge, MergeSpec, Sugar


class ConcatLayer(Merge):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward_multi_input(self, xx, is_training):
        return Z.concat(xx, self.axis)


class ConcatSpec(MergeSpec):
    def __init__(self, axis=1):
        super().__init__()
        assert isinstance(axis, int)
        assert 1 <= axis
        self.axis = axis

    def build_multi_input(self, in_shapes, in_dtypes):
        out_shape = Z.concat_out_shape(in_shapes, self.axis)
        assert len(set(in_dtypes)) == 1
        return ConcatLayer(self.axis), out_shape, in_dtypes[0]


Concat = Sugar(ConcatSpec)
