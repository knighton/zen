from functools import reduce

from ..base import Merge, MergeSpec, Sugar


class MulLayer(Merge):
    def forward_multi_input(self, xx, is_training):
        return reduce(lambda a, b: a * b, xx)


class MulSpec(MergeSpec):
    def build_multi_input(self, in_shapes, in_dtypes):
        assert len(set(in_shapes)) == 1
        assert len(set(in_dtypes)) == 1
        return MulLayer(), in_shapes[0], in_dtypes[0]


Mul = Sugar(MulSpec)
