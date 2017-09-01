def _normalize_axis(axis, num_dims):
    if axis is None:
        return None
    elif isinstance(axis, int):
        axes = [axis]
    elif isinstance(axis, tuple):
        axes = list(axis)
    elif isinstance(axis, list):
        axes = axis
    else:
        assert False
    return list(map(lambda axis: axis % num_dims, axes))


def _reduce_builtin(reduce_func_name, x, axis=None, keep_dims=False):
    axes = _normalize_axis(axis, x.dim())
    if axes is None:
        if not keep_dims:
            reduce_func = getattr(x, reduce_func_name)
            return reduce_func()
        else:
            return _reduce_builtin(reduce_func_name, x, tuple(range(x.dim())),
                                   keep_dims)
    for axis in axes:
        reduce_func = getattr(x, reduce_func_name)
        x = reduce_func(axis, keep_dims)
        if isinstance(x, tuple):
            x = x[0]
    if not keep_dims:
        x = x.squeeze()
    return x


def mean(x, axis=None, keepdims=False):
    return _reduce_builtin('mean', x, axis, keepdims)


def sqrt(x):
    return x.sqrt()


def square(x):
    return x.pow(2)
