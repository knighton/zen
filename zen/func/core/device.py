from .. import backend as Z


def get_num_gpus():
    return Z.get_num_gpus()


def default_device():
    return Z.default_device()


def set_default_device(device):
    return Z.set_default_device(device)


def to_device(x, device=None):
    return Z.to_device(x, device)


def to_cpu(x):
    return Z.to_cpu(x)


def to_gpu(x, device=None):
    return Z.to_gpu(x, device)


set_default_device(get_num_gpus() - 1)
