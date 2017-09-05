import mxnet as mx

from .. import core as C


def _my_constant_pad(x, padding, value):
    ndim = C.get_ndim(x) - 2
    padding = C.normalize_int_padding(padding, ndim, 'padding')
    dtype = C.get_dtype(x)
    load = {
        'constant': C.constant,
        'variable': C.variable,
    }[C.constant_or_variable(x)]
    for dim, (left, right) in enumerate(padding):
        dim += 2
        if not left and not right:
            continue
        parts = []
        if left:
            shape = list(x.size())
            shape[dim] = left
            arr = np.full(shape, value, dtype_)
            parts.append(load(arr))
        parts.append(x)
        if right:
            shape = list(x.size())
            shape[dim] = right
            arr = np.full(shape, value, dtype_)
            parts.append(load(arr))
        x = C.concat(parts, dim)
    return C.cast(x, C.get_dtype(dtype))


constant_pad1d = _my_constant_pad


def constant_pad2d(x, padding, value):
    (top, bottom), (left, right) = padding
    padding = 0, 0, 0, 0, top, bottom, left, right
    ret = mx.nd.pad(x, mode='constant', pad_width=padding, constant_value=value)
    return C.cast(ret, C.get_dtype(x))


def constant_pad3d(x, padding, value):
    (front, back), (top, bottom), (left, right) = padding
    padding = 0, 0, 0, 0, front, back, top, bottom, left, right
    ret = mx.nd.pad(x, mode='constant', pad_width=padding, constant_value=value)
    return C.cast(ret, C.get_dtype(x))


_NDIM2CONSTANT_PAD = {
    1: constant_pad1d,
    2: constant_pad2d,
    3: constant_pad3d,
}


def constant_pad(x, padding, value):
    ndim = C.get_ndim(x) - 2
    return _NDIM2CONSTANT_PAD[ndim](x, padding, value)
