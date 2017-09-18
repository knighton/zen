from torch.nn import functional as F

from ... import core as C


def conv1d(x, kernel, bias, padding, stride, dilation):
    padding = C.to_one(padding)
    stride = C.to_one(stride)
    dilation = C.to_one(dilation)
    return F.conv1d(x, kernel, bias, stride, padding, dilation)


def conv2d(x, kernel, bias, padding, stride, dilation):
    return F.conv2d(x, kernel, bias, stride, padding, dilation)


def conv3d(x, kernel, bias, padding, stride, dilation):
    return F.conv3d(x, kernel, bias, stride, padding, dilation)


_DIM2CONV = {
    1: conv1d,
    2: conv2d,
    3: conv3d,
}


def conv(x, kernel, bias, padding, stride, dilation):
    dim = x.size() - 2
    return _DIM2CONV[dim](x, kernel, bias, padding, stride, dilation)
