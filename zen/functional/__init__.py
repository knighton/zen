import torch
from torch.nn import functional as F


def floatx():
    return 'float32'


def check_dim(dim):
    assert isinstance(dim, int), \
        'Dimension must be an integer (got %s): %s.' % (type(dim), dim)
    assert 1 <= dim, 'Dimension must be positive: %d.' % dim


def dense(x, kernel, bias):
    """
    x       tensor (batch_size, in_channels)
    kernel  tensor (out_channels, in_channels)
    bias    tensor (out_channels,)
    """
    return F.linear(x, kernel, bias)


def sigmoid(x):
    return F.sigmoid(x)


def softmax(x):
    x_shape = x.size()
    tp = x.transpose(1, len(x_shape) - 1)
    tp_shape = tp.size()
    input_2d = tp.contiguous().view(-1, tp_shape[-1])
    _2d = F.softmax(input_2d)
    nd = _2d.view(*tp_shape)
    return nd.transpose(1, len(x_shape) - 1)


def reshape(x, out_shape):
    """
    x          tensor (batch_size, in_shape...)  Data to reshape.
    out_shape  shape...                          Shape without batch_size dim.
    """
    batch_out_shape = (x.size(0),) + out_shape
    return x.view(batch_out_shape)


_EPSILON = 1e-6


def binary_cross_entropy(true, pred):
    pred = pred.clamp(_EPSILON, 1. - _EPSILON)
    return -true * pred.log() - (1. - true) * (1. - pred).log()


def categorical_cross_entropy(true, pred):
    pred = pred.clamp(_EPSILON, 1. - _EPSILON)
    ret = -true * pred.log()
    return ret.mean()


def mean_squared_error(true, pred):
    return (true - pred).pow(2).mean()


def binary_accuracy(true, pred):
    ret = true == pred.round()
    ret = ret.type(torch.cuda.FloatTensor)
    return 100. * ret.mean()


def categorical_accuracy(true, pred):
    true = true.max(1)[1]
    pred = pred.max(1)[1]
    ret = true == pred
    ret = ret.type(torch.cuda.FloatTensor)
    return 100. * ret.mean()
