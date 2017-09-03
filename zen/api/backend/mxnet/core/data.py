import mxnet as mx

from .device import _get_device_context


def get_ndim(x):
    return x.ndim


def get_shape(x):
    return x.shape


def count_params(x):
    return x.size


def constant(arr, dtype=None, device=None):
    dtype = dtype or arr.dtype.name
    ctx = _get_device_context(device)
    return mx.nd.array(arr, ctx, dtype)


def variable(arr, dtype=None, device=None):
    tensor = constant(arr, dtype, device)
    tensor.attach_grad()
    return tensor


def to_numpy(x):
    return x.asnumpy()


def to_scalar(x):
    return x.asscalar()


autograd_record = mx.autograd.record


def update_grad(x, lr):
    x[:] -= lr * x.grad
