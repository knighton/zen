from torch.nn import functional as F

from ... import to_one


def conv1d(x, kernel, bias, padding, stride, dilation):
    padding = to_one(padding)
    stride = to_one(stride)
    dilation = to_one(dilation)
    return F.conv1d(x, kernel, bias, stride, padding, dilation, 1)


def conv2d(x, kernel, bias, padding, stride, dilation):
    return F.conv2d(x, kernel, bias, stride, padding, dilation, 1)


def conv3d(x, kernel, bias, padding, stride, dilation):
    return F.conv3d(x, kernel, bias, stride, padding, dilation, 1)
