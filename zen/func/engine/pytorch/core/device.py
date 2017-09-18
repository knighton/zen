from torch import cuda
from torch.cuda import device_count


def get_num_gpus():
    return device_count()


def default_device():
    return _DEFAULT_DEVICE


def set_default_device(device):
    assert isinstance(device, int)
    global _DEFAULT_DEVICE
    if device == -1:
        _DEFAULT_DEVICE = device
    elif 0 <= device:
        _DEFAULT_DEVICE = device
        cuda.set_device(device)
    else:
        assert False


def to_device(x, device=None):
    if device is None:
        device = default_device()
    if device == -1:
        return x.cpu()
    else:
        return x.cuda(device)


def to_cpu(x):
    return x.cpu()


def to_gpu(x, device=None):
    return x.cuda(device)
