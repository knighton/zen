import mxnet as mx


def reshape(x, out_shape):
    batch_out_shape = (x.shape[0],) + out_shape
    return mx.nd.reshape(x, batch_out_shape)
