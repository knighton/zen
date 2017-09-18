from torch.nn import functional as F


def elu(x, alpha):
    return F.elu(x, alpha)


def hard_shrink(x, lambda_):
    return F.hardshrink(x, lambda_)


def leaky_relu(x, alpha):
    return F.leaky_relu(x, alpha)


def selu(x):
    return F.selu(x)


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


def softshrink(x):
    return F.softshrink(x)


def softsign(x):
    return F.softsign(x)


def tanh_shrink(x):
    return F.tanhshrink(x)
