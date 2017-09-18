import mxnet as mx
import os


def num_gpus():
    i = 0
    while True:
        python = 'import mxnet as mx; mx.nd.zeros((1,), ctx=mx.gpu(%d))' % i
        cmd = 'python3 -c "%s" 2> /dev/null' % python
        if os.system(cmd):
            return i
        i += 1


def default_device():
    return _DEFAULT_DEVICE


def set_default_device(device):
    assert isinstance(device, int)
    assert -1 <= device
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = device


def to_device(x, device=None):
    if device is None:
        device = default_device()
    if device == -1:
        return x.as_in_context(mx.cpu())
    else:
        return x.as_in_context(mx.gpu(device))


def _get_device_context(device=None):
    if device is None:
        device = default_device()
    if device == -1:
        return mx.cpu()
    else:
        return mx.gpu(device)
