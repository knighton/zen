import numpy as np

from ... import api as Z
from ..base import Transform, TransformSpec, Sugar


class ReshapeLayer(Transform):
    def __init__(self, out_shape):
        super().__init__()
        self.out_shape = out_shape

    def forward(self, x, is_training):
        return Z.reshape(x, self.out_shape)


class ReshapeSpec(TransformSpec):
    def __init__(self, out_shape):
        super().__init__()
        wildcard_count = 0
        wildcard_axis = None
        numel = 1
        for axis, dim in enumerate(out_shape):
            if dim == -1:
                assert not wildcard_count
                wildcard_count += 1
                wildcard_axis = axis
            else:
                assert isinstance(dim, int)
                assert 0 <= dim
                numel *= dim
        self.out_shape = out_shape
        self.wildcard_axis = wildcard_axis
        self.out_numel = numel

    def build(self, in_shape, in_dtype):
        in_numel = int(np.prod(in_shape))
        if self.wildcard_axis is None:
            assert in_numel == self.out_numel
            out_shape = self.out_shape
        else:
            assert not in_numel % self.out_numel
            out_shape = list(self.out_shape)
            out_shape[self.wildcard_axis] = in_numel // self.out_numel
            out_shape = tuple(out_shape)
        return ReshapeLayer(out_shape), out_shape, in_dtype


Flatten = Sugar(ReshapeSpec, {'out_shape': (-1,)})
Reshape = Sugar(ReshapeSpec)
