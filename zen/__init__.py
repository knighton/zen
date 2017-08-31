import numpy as np
import torch
from torch.autograd import Variable


def variable(arr, tensor_type=torch.cuda.FloatTensor):
    tensor = torch.from_numpy(arr).type(tensor_type)
    return Variable(tensor, requires_grad=True)


def constant(arr, tensor_type=torch.cuda.FloatTensor):
    tensor = torch.from_numpy(arr).type(tensor_type)
    return Variable(tensor, requires_grad=False)


def init(*shape):
    return np.random.normal(0., 1., shape)
