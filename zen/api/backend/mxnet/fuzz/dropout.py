import mxnet as mx

from ..core.device import _get_device_context


def dropout(x, rate, is_training):
    data = mx.sym.Variable('data')
    dropout = mx.sym.Dropout(data, p=rate)
    executor = dropout.simple_bind(data=x.shape, ctx=_get_device_context())
    executor.forward(data=x, is_train=is_training)
    return executor.outputs[0]


dropout0d = dropout
dropout1d = dropout
dropout2d = dropout
dropout3d = dropout
