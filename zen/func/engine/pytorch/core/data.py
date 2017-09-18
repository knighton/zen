from contextlib import contextmanager
import torch
from torch.autograd import Variable
from torch import _TensorBase

from .device import to_device
from .type import cast


def ndim(x):
    return x.dim()


def shape(x):
    return tuple(x.size())


def permute(x, axes):
    return x.permute(*axes)


def repeat(x, shape):
    return x.repeat(*shape)


def expand_dims(x, axis):
    return x.unsqueeze(axis)


def squeeze(x, axis=None):
    return x.squeeze(axis)


def concat(xx, axis):
    return torch.cat(xx, axis)


def stack(xx, axis=0):
    return torch.stack(xx, axis)


def size(x):
    return x.numel()


def clone(x):
    return x.clone()


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


def constant_or_variable(x):
    assert isinstance(x, Variable)
    return 'constant' if x.grad is None else 'variable'


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


def backward(loss_variables, grad_tensors):
    torch.autograd.backward(loss_variables, grad_tensors)


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
