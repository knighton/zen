import sys
import torch
from torch.nn import functional as F

from .fuzz import *


def get(name, dim=None):
    if dim is not None:
        name = '%s%dd' % (name, dim)
    module = sys.modules[__name__]
    return getattr(module, name)


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


def to_one(x):
    if isinstance(x, int):
        return x
    elif isinstance(x, tuple):
        assert len(x) == 1
        return x[0]
    else:
        assert False


def to_shape(x, dim):
    if isinstance(x, int):
        return (x,) * dim
    elif isinstance(x, tuple):
        assert len(x) == dim
        return x
    else:
        assert False


def conv1d(x, kernel, bias, padding, stride, dilation):
    """
    x         tensor (batch_size, in_channels, width)
    kernel    tensor (out_channels, in_channels, width)
    bias      tensor (out_channels,)
    padding   dim or shape
    stride    dim or shape
    dilation  dim or shape
    """
    padding = to_one(padding)
    stride = to_one(stride)
    dilation = to_one(dilation)
    return F.conv1d(x, kernel, bias, stride, padding, dilation, 1)


def conv2d(x, kernel, bias, padding, stride, dilation):
    """
    x         tensor (batch_size, in_channels, height, width)
    kernel    tensor (out_channels, in_channels, height, width)
    bias      tensor (out_channels,)
    padding   dim or shape
    stride    dim or shape
    dilation  dim or shape
    """
    return F.conv2d(x, kernel, bias, stride, padding, dilation, 1)


def conv3d(x, kernel, bias, padding, stride, dilation):
    """
    x         tensor (batch_size, in_channels, depth, height, width)
    kernel    tensor (out_channels, in_channels, depth, height, width)
    bias      tensor (out_channels,)
    padding   dim or shape
    stride    dim or shape
    dilation  dim or shape
    """
    return F.conv3d(x, kernel, bias, stride, padding, dilation, 1) 


def conv_out_shape(in_shape, filter_shape, padding, stride, dilation):
    """
    in_shape      shape... (of x without channel dim)
    filter_shape  dim or shape
    padding       dim or shape
    stride        dim or shape
    dilation      dim or shape
    """
    dim = len(in_shape)
    filter_shape = to_shape(filter_shape, dim)
    padding = to_shape(padding, dim)
    stride = to_shape(stride, dim)
    dilation = to_shape(dilation, dim)
    out_shape = [None] * len(in_shape)
    for i in range(len(in_shape)):
        numerator = in_shape[i] + 2 * padding[i] - \
                    dilation[i] * (filter_shape[i] - 1) - 1
        out_shape[i] = floor(numerator / stride[i] + 1)
    return tuple(out_shape)


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


def binary_cross_entropy(true, pred):
    pred = pred.clamp(epsilon(), 1. - epsilon())
    return -true * pred.log() - (1. - true) * (1. - pred).log()


def categorical_cross_entropy(true, pred):
    pred = pred.clamp(epsilon(), 1. - epsilon())
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
