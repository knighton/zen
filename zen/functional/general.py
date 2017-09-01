from torch.nn import functional as F

from .core import mean, sqrt, square


def _do_batch_norm(x, mean, variance, beta, gamma):
    return gamma * ((x - mean) / Z.sqrt(variance + epsilon())) + beta


def _moments(x, axes):
    shift = Z.mean(x, axes, True)
    shifted_mean = Z.mean(x - shift, axes, True)
    variance_mean = Z.mean(Z.square(x - shift), axes, True)
    variance = variance_mean - Z.square(shifted_mean)
    mean = shifted_mean + shift
    return mean, variance


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
