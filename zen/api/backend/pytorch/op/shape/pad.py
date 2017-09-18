from torch.nn import functional as F

from ... import core as C


def constant_pad1d(x, padding, value):
    x = x.unsqueeze(2)
    (left, right), = C.normalize_int_padding(padding, 2, 'padding')
    padding = 0, 0, left, right
    x = constant_pad2d(x, padding, value)
    return x.squeeze(2)


def constant_pad2d(x, padding, value):
    (top, bottom), (left, right) = \
        C.normalize_int_padding(padding, 2, 'padding')
    padding = top, bottom, left, right
    return F.pad(x, padding, 'constant', value)


def constant_pad3d(x, padding, value):
    (front, back), (top, bottom), (left, right) = \
        C.normalize_int_padding(padding, 3, 'padding')
    padding = front, back, top, bottom, left, right
    return F.pad(x, padding, 'constant', value)


_NDIM2CONSTANT_PAD = {
    1: constant_pad1d,
    2: constant_pad2d,
    3: constant_pad3d,
}


def constant_pad(x, padding, value):
    ndim = C.get_ndim(x) - 2
    return _NDIM2CONSTANT_PAD[ndim](x, padding, value)


def edge_pad1d(x, padding):
    x = x.unsqueeze(2)
    (left, right), = C.normalize_int_padding(padding, 2, 'padding')
    padding = 0, 0, left, right
    x = edge_pad2d(x, padding)
    return x.squeeze(2)


def edge_pad2d(x, padding, value):
    (top, bottom), (left, right) = \
        C.normalize_int_padding(padding, 2, 'padding')
    padding = top, bottom, left, right
    return F.pad(x, padding, 'replicate')


def edge_pad3d(x, padding, value):
    (front, back), (top, bottom), (left, right) = \
        C.normalize_int_padding(padding, 3, 'padding')
    padding = front, back, top, bottom, left, right
    return F.pad(x, padding, 'replicate')


_NDIM2EDGE_PAD = {
    1: edge_pad1d,
    2: edge_pad2d,
    3: edge_pad3d,
}


def edge_pad(x, padding):
    ndim = C.get_ndim(x) - 2
    return _NDIM2EDGE_PAD[ndim](x, padding)


def reflect_pad1d(x, padding):
    x = x.unsqueeze(2)
    (left, right), = C.normalize_int_padding(padding, 2, 'padding')
    padding = 0, 0, left, right
    x = reflect_pad2d(x, padding)
    return x.squeeze(2)


def reflect_pad2d(x, padding, value):
    (top, bottom), (left, right) = \
        C.normalize_int_padding(padding, 2, 'padding')
    padding = top, bottom, left, right
    return F.pad(x, padding, 'reflect')


def reflect_pad3d(x, padding, value):
    (front, back), (top, bottom), (left, right) = \
        C.normalize_int_padding(padding, 3, 'padding')
    padding = front, back, top, bottom, left, right
    return F.pad(x, padding, 'reflect')


_NDIM2REFLECT_PAD = {
    1: reflect_pad1d,
    2: reflect_pad2d,
    3: reflect_pad3d,
}


def reflect_pad(x, padding):
    ndim = C.get_ndim(x) - 2
    return _NDIM2REFLECT_PAD[ndim](x, padding)
