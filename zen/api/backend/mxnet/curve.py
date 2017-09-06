import mxnet as mx


def elu(x, alpha=1.):
    return mx.nd.LeakyReLU(x, 'elu', alpha)


def leaky_relu(xi, alpha=0.1):
    return mx.nd.LeakyReLU(x, 'leaky', alpha)


def sigmoid(x):
    return mx.nd.sigmoid(x)


def softmax(x):
    return mx.nd.softmax(x)


def softplus(x):
    return mx.nd.Activation(x, 'softrelu')


def tanh(x):
    return mx.nd.tanh(x)
