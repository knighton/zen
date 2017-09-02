import torch
from torch.autograd import Variable


def get_ndim(x):
    return x.dim()


def get_shape(x):
    return tuple(x.size())


def to_variable(arr, tensor_type=torch.cuda.FloatTensor):
    tensor = torch.from_numpy(arr).type(tensor_type)
    return Variable(tensor, requires_grad=True)


def to_constant(arr, tensor_type=torch.cuda.FloatTensor):
    tensor = torch.from_numpy(arr).type(tensor_type)
    return Variable(tensor, requires_grad=False)


def update_grad(param, lr):
    param.data -= lr * param.grad.data
    param.grad.data.zero_()
