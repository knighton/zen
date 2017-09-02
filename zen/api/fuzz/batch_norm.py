from .. import backend as Z
from .. import core as C


def _do_batch_norm(x, mean, variance, beta, gamma):
    return gamma * ((x - mean) / C.sqrt(variance + C.epsilon())) + beta


def _moments(x, axes):
    shift = C.mean(x, axes, True)
    shifted_mean = C.mean(x - shift, axes, True)
    variance_mean = C.mean(C.square(x - shift), axes, True)
    variance = variance_mean - C.square(shifted_mean)
    mean_ = shifted_mean + shift
    return mean_, variance


def _running_average_update(x_running, x_new, momentum):
    x_running.data = momentum * x_running.data + (1. - momentum) * x_new.data


def _my_batch_norm(x, is_training, reduction_axes, momentum, beta, gamma,
                   running_mean, running_variance):
    if is_training:
        mean, variance = _moments(x, reduction_axes)
        x = _do_batch_norm(x, mean, variance, beta, gamma)
        _running_average_update(running_mean, mean, momentum)
        _running_average_update(running_variance, variance, momentum)
    else:
        x = _do_batch_norm(x, running_mean, running_variance, beta, gamma)
    return x


batch_norm = Z.get('batch_norm', _my_batch_norm)
batch_norm0d = Z.get('batch_norm0d', _my_batch_norm)
batch_norm1d = Z.get('batch_norm1d', _my_batch_norm)
batch_norm2d = Z.get('batch_norm2d', _my_batch_norm)
batch_norm3d = Z.get('batch_norm3d', _my_batch_norm)
