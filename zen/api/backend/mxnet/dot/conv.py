import mxnet as mx


def conv(x, kernel, bias, padding, stride, dilation):
    return mx.nd.Convolution(
        x, kernel, bias, kernel.shape[2:], stride, dilation, padding,
        kernel.shape[0])


conv1d = conv
conv2d = conv
conv3d = conv
