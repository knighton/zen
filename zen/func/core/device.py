from .. import engine as E


def get_num_gpus():
    return E.get_num_gpus()


def default_device():
    return E.default_device()


def set_default_device(device):
    return E.set_default_device(device)


def to_device(x, device=None):
    return E.to_device(x, device)


def to_cpu(x):
    return E.to_cpu(x)


def to_gpu(x, device=None):
    return E.to_gpu(x, device)


set_default_device(get_num_gpus() - 1)
