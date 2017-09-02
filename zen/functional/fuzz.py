import numpy as np
from torch.nn import functional as F

from .core import epsilon, mean, sqrt, square

from .. import constant


to_constant = constant


def shape(x):
    return tuple(x.size())


def _do_batch_norm(x, mean, variance, beta, gamma):
    return gamma * ((x - mean) / sqrt(variance + epsilon())) + beta


def _moments(x, axes):
    shift = mean(x, axes, True)
    shifted_mean = mean(x - shift, axes, True)
    variance_mean = mean(square(x - shift), axes, True)
    variance = variance_mean - square(shifted_mean)
    mean_ = shifted_mean + shift
    return mean_, variance


def _running_average_update(x_running, x_new, momentum):
    x_running.data = momentum * x_running.data + (1. - momentum) * x_new.data


def my_batch_norm(x, is_training, reduction_axes, momentum, beta, gamma,
               running_mean, running_variance):
    if is_training:
        mean, variance = _moments(x, reduction_axes)
        x = _do_batch_norm(x, mean, variance, beta, gamma)
        _running_average_update(running_mean, mean, momentum)
        _running_average_update(running_variance, variance, momentum)
    else:
        x = _do_batch_norm(x, running_mean, running_variance, beta, gamma)
    return x


def batch_norm(x, is_training, reduction_axes, momentum, beta, gamma,
               running_mean, running_variance):
    running_mean = running_mean.squeeze().data
    running_variance = running_variance.squeeze().data
    gamma = gamma.squeeze()
    beta = beta.squeeze()
    return F.batch_norm(x, running_mean, running_variance, gamma, beta,
                        is_training, momentum, 1e-3)


batch_norm0d = batch_norm
batch_norm1d = batch_norm
batch_norm2d = batch_norm
batch_norm3d = batch_norm


def dropout(x, is_training, rate):
    return F.dropout(x, rate, is_training, False)


dropout0d = dropout
dropout1d = dropout
dropout2d = dropout
dropout3d = dropout


def spatial_dropout(x, is_training, rate):
    if not is_training:
        return x

    shape = shape(x)
    noise_shape = shape[:2] + (1,) * (len(shape) - 2)
    max_value = 1. / rate
    mask = np.random.uniform(0, max_value, noise_shape).astype('float32')
    mask = np.floor(mask.clip(0., 1.))
    mask = to_constant(mask)
    return x * mask / (1. - rate)


spatial_dropout1d = spatial_dropout


def spatial_dropout2d(x, is_training, rate):
    return F.dropout2d(x, rate, is_training, False)


def spatial_dropout3d(x, is_training, rate):
    return F.dropout3d(x, rate, is_training, False)
