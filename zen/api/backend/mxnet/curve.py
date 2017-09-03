import mxnet as mx


def sigmoid(x):
    return mx.nd.sigmoid(x)


def softmax(x):
    return mx.nd.softmax(x)
