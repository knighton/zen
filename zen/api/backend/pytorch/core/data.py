from contextlib import contextmanager
import torch
from torch.autograd import Variable
from torch import _TensorBase

from .device import to_device
from .dtype import cast


def get_ndim(x):
    return x.dim()


def get_shape(x):
    return tuple(x.size())


def count_params(x):
    return x.numel()


def tensor(arr, dtype=None, device=None):
    dtype = dtype or arr.dtype.name
    x = torch.from_numpy(arr)
    x = cast(x, dtype)
    x = to_device(x, device)
    return x


def constant(arr, dtype=None, device=None):
    tensor_ = tensor(arr, dtype, device)
    return Variable(tensor_, requires_grad=False)


def variable(arr, dtype=None, device=None):
    tensor_ = tensor(arr, dtype, device)
    return Variable(tensor_, requires_grad=True)


def to_numpy(x):
    if isinstance(x, Variable):
        tensor = x.data
    elif isinstance(x, _TensorBase):
        tensor = x
    else:
        assert False
    return tensor.cpu().numpy()


def to_scalar(x):
    assert x.numel() == 1
    return to_numpy(x).flatten()[0]


@contextmanager
def autograd_record():
    yield


def data(x):
    return x.data


def gradient(x):
    return x.grad.data


def update(x, new_value):
    x.data[:] = new_value
    if x.grad.volatile:
        x.grad.data.zero_()
    else:
        data = x.grad.data
        x.grad = Variable(data.new().resize_as_(data).zero_())
