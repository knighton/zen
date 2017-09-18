from ... import func as Z
from ..base import Merge, MergeSpec, Transform, TransformSpec, Sugar


class EachPairLayer(Transform):
    def __init__(self, ndim, relater, global_pool):
        super().__init__()
        self.each_pair = Z.get('each_pair', ndim)
        self.relater = relater
        self.global_pool = global_pool
        for param in relater.model_params():
            self.add_variable_param(param)

    def forward(self, x, is_training):
        return self.each_pair(x, None, is_training, self.relater,
                              self.global_pool)


class EachPairSpec(TransformSpec):
    def __init__(self, relater, global_pool=False, ndim=None):
        from ...model.model import Model
        super().__init__()
        assert isinstance(relater, Model)
        assert isinstance(global_pool, bool)
        Z.check_input_ndim(ndim, {1, 2, 3}, 'ndim')
        self.relater = relater
        self.global_pool = global_pool
        self.ndim = ndim

    def build(self, in_shape, in_dtype):
        ndim = Z.verify_input_ndim(self.ndim, in_shape)
        layer = EachPairLayer(ndim, self.relater, self.global_pool)
        out_shape = Z.each_pair_out_shape(in_shape, None, self.relater,
                                          self.global_pool)
        return layer, out_shape, in_dtype


EachPair = Sugar(EachPairSpec)
EachPair1D = Sugar(EachPairSpec, {'ndim': 1})
EachPair2D = Sugar(EachPairSpec, {'ndim': 2})
EachPair3D = Sugar(EachPairSpec, {'ndim': 3})


class EachPairWithLayer(Merge):
    def __init__(self, ndim, relater, global_pool):
        super().__init__()
        self.each_pair = Z.get('each_pair', ndim)
        self.relater = relater
        self.global_pool = global_pool
        for param in relater.model_params():
            self.add_variable_param(param)

    def forward_multi_input(self, xx, is_training):
        grid, concat_to_each = xx
        return self.each_pair(grid, concat_to_each, is_training, self.relater,
                              self.global_pool)


class EachPairWithSpec(MergeSpec):
    def __init__(self, relater, global_pool=False, ndim=None):
        from ...model.model import Model
        super().__init__()
        assert isinstance(relater, Model)
        assert isinstance(global_pool, bool)
        Z.check_input_ndim(ndim, {1, 2, 3}, 'ndim')
        self.relater = relater
        self.global_pool = global_pool
        self.ndim = ndim

    def build_multi_input(self, in_shapes, in_dtypes):
        assert len(in_shapes) == 2
        assert len(set(in_dtypes)) == 1
        grid_shape, concat_to_each_shape = in_shapes
        ndim = Z.verify_input_ndim(self.ndim, grid_shape)
        Z.check_shape(concat_to_each_shape, 1, 'concat_to_each_shape')
        layer = EachPairWithLayer(ndim, self.relater, self.global_pool)
        out_shape = Z.each_pair_out_shape(
            grid_shape, concat_to_each_shape, self.relater, self.global_pool)
        return layer, out_shape, in_dtypes[0]


EachPairWith = Sugar(EachPairWithSpec)
EachPairWith1D = Sugar(EachPairWithSpec, {'ndim': 1})
EachPairWith2D = Sugar(EachPairWithSpec, {'ndim': 2})
EachPairWith3D = Sugar(EachPairWithSpec, {'ndim': 3})
