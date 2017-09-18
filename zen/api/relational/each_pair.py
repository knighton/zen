import numpy as np

from .. import core as C


def _get_slices(index, spatial_grid_shape):
    indexes = []
    for dim in spatial_grid_shape:
        indexes.append(index % dim)
        index //= dim
    slices = list(map(lambda index: slice(index, index + 1), indexes))
    slices = [slice(None)] + slices + [slice(None)]
    return tuple(slices)


def each_pair(grid, concat_to_each, is_training, relater, flat):
    # Get shapes.
    grid_shape = C.get_shape(grid)
    batch_size, num_grid_channels = grid_shape[:2]
    spatial_shape = grid_shape[2:]
    num_cells = int(np.prod(spatial_shape))

    # Flatten `grid` shapewise and put the channels dimension last.
    grid = C.reshape(grid, (batch_size, num_grid_channels, num_cells))
    grid = C.permute(grid, (0, 2, 1))

    # Repeat for concatenation.
    left = C.expand_dims(grid, 1)
    left = C.repeat(left, (1, num_cells, 1, 1))
    right = C.expand_dims(grid, 2)
    right = C.repeat(right, (1, 1, num_cells, 1))

    # Do the concat.
    if concat_to_each is None:
        grid_x_grid = C.concat([left, right], 3)
    else:
        concat_to_each = C.expand_dims(concat_to_each, 1)
        concat_to_each = C.repeat(concat_to_each, (1, num_cells, 1))
        concat_to_each = C.expand_dims(concat_to_each, 2)
        concat_to_each = C.repeat(concat_to_each, (1, 1, num_cells, 1))
        grid_x_grid = C.concat([left, right, concat_to_each], 3)

    # Reshape to pass the tuples through `relater`.
    relater_in = C.reshape(grid_x_grid, (batch_size * num_cells ** 2, -1))

    # Relate each tuple of embeddings.
    relater_outs = relater.model_forward([relater_in], is_training)
    assert len(relater_outs) == 1
    relater_out, = relater_outs

    if flat:
        # Sum the outputs, resulting in a single 'globally pooled' embedding.
        ret = C.reshape(relater_out, (batch_size, num_cells ** 2, -1))
        ret = C.sum(ret, 1)
    else:
        # For each cell (depth x height x width), sum all embeddings that
        # involve it, preserving the input's spatial shape.

        # Break it down into the actual dimensions (4, 6, or 8).
        true_shape = (batch_size,) + spatial_shape + spatial_shape + (-1,)
        ret = C.reshape(relater_out, true_shape)

        # Permute the depth, height, and width dimensions together.
        permute_axes = [0, len(true_shape) - 1]
        for i in range(len(spatial_shape)):
            permute_axes.append(i + 1)  # Left embedding.
            permute_axes.append(i + 1 + len(spatial_shape))  # Right embedding.
        ret = C.permute(ret, permute_axes)

        # Reduce over depth, height, and width.
        for i in reversed(range(len(spatial_shape))):
            ret = C.sum(ret, (i + 1) * 2)
    return ret


each_pair1d = each_pair
each_pair2d = each_pair
each_pair3d = each_pair


def each_pair_out_shape(grid_shape, concat_to_each_shape, relater, flat):
    if concat_to_each_shape is None:
        dim = 2 * grid_shape[0]
    else:
        dim = 2 * grid_shape[0] + concat_to_each_shape[0]
    relater_in_shape = (1, dim)
    arr = np.zeros(relater_in_shape, dtype=C.floatx())
    relater_outs = relater.predict_on_batch([arr])
    relater_out, = relater_outs
    relater_out_shape = relater_out.shape[1:]
    if flat:
        ret = relater_out_shape
    else:
        shape = list(grid_shape)
        shape[0] = relater_out_shape[0]
        ret = tuple(shape)
    return ret
