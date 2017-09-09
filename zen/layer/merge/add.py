from ..base import Merge, MergeSpec, Sugar


class AddLayer(Merge):
    def forward_multi_input(self, xx, is_training):
        return sum(xx)


class AddSpec(MergeSpec):
    def build_multi_input(self, in_shapes, in_dtypes):
        assert len(set(in_shapes)) == 1
        assert len(set(in_dtypes)) == 1
        return AddLayer(), in_shapes[0], in_dtypes[0]


Add = Sugar(AddSpec)
