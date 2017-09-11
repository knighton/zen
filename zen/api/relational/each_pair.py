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
    spatial_grid_shape = grid_shape[2:]
    num_cells = int(np.prod(spatial_grid_shape))

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
        # Sum the outputs.
        ret = C.reshape(relater_out, (batch_size, num_cells ** 2, -1))
        ret = C.sum(ret, 1)
    else:
        # Add every cell's embedding per depth, height, and width, preserving
        # the input's spatial shape.
        relater_out = C.reshape(relater_out, (batch_size, num_cells ** 2, -1))
        _, _2, y_channels = C.get_shape(relater_out)
        distributed_shape = (batch_size,) + spatial_grid_shape + (y_channels,)
        distributed = C.variable(np.zeros(distributed_shape, dtype=C.floatx()))
        for left in range(num_cells):
            lefts_slices = _get_slices(left, spatial_grid_shape)
            for right in range(num_cells):
                rights_slices = _get_slices(right, spatial_grid_shape)
                embedding = relater_out[:, left * num_cells + right, :]
                for i in range(1, len(spatial_grid_shape)):
                    embedding = C.expand_dims(embedding, 1)
                distributed[lefts_slices] += embedding
                distributed[rights_slices] += embedding
        permute_shape = (batch_size,) + (y_channels,) + spatial_grid_shape
        ret = C.permute(distributed, permute_shape)
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
