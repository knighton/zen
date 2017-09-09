from .. import core as C


concat = C.concat


def concat_out_shape(in_shapes, axis):
    """
    in_shapes   (in_channels, shape...)
    axis        int
    """
    assert isinstance(axis, int)
    assert 1 <= axis
    axis -= 1
    shapes = set()
    concat_dim = 0
    for shape in in_shapes:
        shape = list(shape)
        concat_dim += shape[axis]
        shape[axis] = None
        shape = tuple(shape)
        shapes.add(shape)
    assert len(shapes) == 1
    out_shape = list(list(shapes)[0])
    out_shape[axis] = concat_dim
    return tuple(out_shape)
