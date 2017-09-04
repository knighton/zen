import mxnet as mx

from .device import _get_device_context


def get_ndim(x):
    return x.ndim


def get_shape(x):
    return x.shape


def expand_dims(x, axis):
    return mx.nd.expand_dims(x, axis)


def concat(xx, axis):
    return mx.nd.concat(*xx, dim=axis)


def squeeze(x, axis):
    shape = list(x.shape)
    axis %= len(shape)
    assert shape[axis] == 1, \
        'Tried to squeeze axis %d of shape %s.' % (axis, x.shape)
    new_shape = tuple(shape[:axis] + shape[axis + 1:])
    return mx.nd.reshape(new_shape)


def count_params(x):
    return x.size


def tensor(arr, dtype=None, device=None):
    dtype = dtype or arr.dtype.name
    ctx = _get_device_context(device)
    return mx.nd.array(arr, ctx, dtype)


constant = tensor


def variable(arr, dtype=None, device=None):
    ret = tensor(arr, dtype, device)
    ret.attach_grad()
    return ret


def to_numpy(x):
    return x.asnumpy()


def to_scalar(x):
    return x.asscalar()


autograd_record = mx.autograd.record


def data(x):
    return x


def gradient(x):
    return x.grad


def update(x, new_value):
    x = new_value
    x.grad[:] = 0
