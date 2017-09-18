import mxnet as mx

from ... import core as C


_BATCH_CHANNELS_PADDING = 0, 0, 0, 0


def constant_pad1d(x, padding, value):
    x = mx.nd.expand_dims(x, 2)
    (left, right), = C.normalize_int_padding(padding, 2, 'padding')
    padding = 0, 0, left, right
    x = constant_pad2d(x, padding, value)
    return C.squeeze(x, 2)


def constant_pad2d(x, padding, value):
    (top, bottom), (left, right) = padding
    padding = _BATCH_CHANNELS_PADDING + (top, bottom, left, right)
    ret = mx.nd.pad(x, mode='constant', pad_width=padding, constant_value=value)
    return C.cast(ret, C.dtype(x))


def constant_pad3d(x, padding, value):
    (front, back), (top, bottom), (left, right) = padding
    padding = _BATCH_CHANNELS_PADDING + (front, back, top, bottom, left, right)
    ret = mx.nd.pad(x, mode='constant', pad_width=padding, constant_value=value)
    return C.cast(ret, C.dtype(x))


_NDIM2CONSTANT_PAD = {
    1: constant_pad1d,
    2: constant_pad2d,
    3: constant_pad3d,
}


def constant_pad(x, padding, value):
    ndim = C.ndim(x) - 2
    return _NDIM2CONSTANT_PAD[ndim](x, padding, value)


def edge_pad1d(x, padding):
    x = mx.nd.expand_dims(x, 2)
    (left, right), = C.normalize_int_padding(padding, 2, 'padding')
    padding = 0, 0, left, right
    x = edge_pad2d(x, padding)
    return C.squeeze(x, 2)


def edge_pad2d(x, padding, value):
    (top, bottom), (left, right) = \
        C.normalize_int_padding(padding, 2, 'padding')
    padding = _BATCH_CHANNELS_PADDING + (top, bottom, left, right)
    ret = mx.nd.pad(x, mode='edge', pad_width=padding)
    return C.cast(ret, C.dtype(x))


def edge_pad3d(x, padding, value):
    (front, back), (top, bottom), (left, right) = \
        C.normalize_int_padding(padding, 3, 'padding')
    padding = _BATCH_CHANNELS_PADDING + (front, back, top, bottom, left, right)
    ret = mx.nd.pad(x, mode='edge', pad_width=padding)
    return C.cast(ret, C.dtype(x))


_NDIM2EDGE_PAD = {
    1: edge_pad1d,
    2: edge_pad2d,
    3: edge_pad3d,
}


def edge_pad(x, padding):
    ndim = C.ndim(x) - 2
    return _NDIM2EDGE_PAD[ndim](x, padding)


def reflect_pad1d(x, padding):
    x = mx.nd.expand_dims(x, 2)
    (left, right), = C.normalize_int_padding(padding, 2, 'padding')
    padding = 0, 0, left, right
    x = reflect_pad2d(x, padding)
    return C.squeeze(x, 2)


def reflect_pad2d(x, padding, value):
    (top, bottom), (left, right) = \
        C.normalize_int_padding(padding, 2, 'padding')
    padding = _BATCH_CHANNELS_PADDING + (top, bottom, left, right)
    ret = mx.nd.pad(x, mode='reflect', pad_width=padding)
    return C.cast(ret, C.dtype(x))


def reflect_pad3d(x, padding, value):
    (front, back), (top, bottom), (left, right) = \
        C.normalize_int_padding(padding, 3, 'padding')
    padding = _BATCH_CHANNELS_PADDING + (front, back, top, bottom, left, right)
    ret = mx.nd.pad(x, mode='reflect', pad_width=padding)
    return C.cast(ret, C.dtype(x))


_NDIM2REFLECT_PAD = {
    1: reflect_pad1d,
    2: reflect_pad2d,
    3: reflect_pad3d,
}


def reflect_pad(x, padding):
    ndim = C.ndim(x) - 2
    return _NDIM2REFLECT_PAD[ndim](x, padding)
