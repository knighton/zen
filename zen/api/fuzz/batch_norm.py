from .. import backend as Z
from .. import core as C


def _do_batch_norm(x, mean, variance, beta, gamma):
    """
    Actually do the work.
    """
    return gamma * ((x - mean) / C.sqrt(variance + C.epsilon())) + beta


def _moments(x, axes):
    """
    Get mean and variance.
    """
    shift = C.mean(x, axes, True)
    shifted_mean = C.mean(x - shift, axes, True)
    variance_mean = C.mean(C.square(x - shift), axes, True)
    variance = variance_mean - C.square(shifted_mean)
    mean = shifted_mean + shift
    return mean, variance


def _running_average_update(x_running, x_new, momentum):
    """
    Update a moving average.
    """
    x_running.data = momentum * x_running.data + (1. - momentum) * x_new.data


def _my_batch_norm(x, is_training, reduction_axes, momentum, beta, gamma,
                   running_mean, running_variance):
    """
    Non-built-in batch normalization.
    """
    if is_training:
        mean, variance = _moments(x, reduction_axes)
        x = _do_batch_norm(x, mean, variance, beta, gamma)
        _running_average_update(running_mean, mean, momentum)
        _running_average_update(running_variance, variance, momentum)
    else:
        x = _do_batch_norm(x, running_mean, running_variance, beta, gamma)
    return x


"""
ND batch normalization.

Input:
    x                 variable (batch_size, channels,  The input.
                                shape...)
    is_training       bool                             Whether in training mode.
    reduction_axes    list of dimensions (ints)        Axes to normalize over.
    momentum          0. <= float <= 1.                How quickly to update the
                                                       running statistics.
    beta              variable (shape is x shape with  Batch norm beta.
                      reduction_axes dims set to 1)
    gamma             variable (shape ")               Batch norm gamma.
    running_mean      constant (shape ")               Batch norm mean.
    running_variance  constant (shape ")               Batch norm variance.

Output:
    y                 variable (batch_size, channels,  The output.
                                shape...)
"""
batch_norm = Z.get('batch_norm', _my_batch_norm)


"""
0D batch normalization.

Input:
    x                 variable (batch_size, channels)  The input.
    is_training       bool                             Whether in training mode.
    reduction_axes    list of dimensions (ints)        Axes to normalize over.
    momentum          0. <= float <= 1.                How quickly to update the
                                                       running statistics.
    beta              variable (shape is x shape with  Batch norm beta.
                      reduction_axes dims set to 1)
    gamma             variable (shape ")               Batch norm gamma.
    running_mean      constant (shape ")               Batch norm mean.
    running_variance  constant (shape ")               Batch norm variance.

Output:
    y                 variable (batch_size, channels,  The output.
                                shape...)
"""
batch_norm0d = Z.get('batch_norm0d', _my_batch_norm)


"""
1D batch normalization.

Input:
    x                 variable (batch_size, channels,  The input.
                                width)
    is_training       bool                             Whether in training mode.
    reduction_axes    list of dimensions (ints)        Axes to normalize over.
    momentum          0. <= float <= 1.                How quickly to update the
                                                       running statistics.
    beta              variable (shape is x shape with  Batch norm beta.
                      reduction_axes dims set to 1)
    gamma             variable (shape ")               Batch norm gamma.
    running_mean      constant (shape ")               Batch norm mean.
    running_variance  constant (shape ")               Batch norm variance.

Output:
    y                 variable (batch_size, channels,  The output.
                                shape...)
"""
batch_norm1d = Z.get('batch_norm1d', _my_batch_norm)


"""
2D batch normalization.

Input:
    x                 variable (batch_size, channels,  The input.
                                height, width)
    is_training       bool                             Whether in training mode.
    reduction_axes    list of dimensions (ints)        Axes to normalize over.
    momentum          0. <= float <= 1.                How quickly to update the
                                                       running statistics.
    beta              variable (shape is x shape with  Batch norm beta.
                      reduction_axes dims set to 1)
    gamma             variable (shape ")               Batch norm gamma.
    running_mean      constant (shape ")               Batch norm mean.
    running_variance  constant (shape ")               Batch norm variance.

Output:
    y                 variable (batch_size, channels,  The output.
                                shape...)
"""
batch_norm2d = Z.get('batch_norm2d', _my_batch_norm)



"""
3D batch normalization.

Input:
    x                 variable (batch_size, channels,  The input.
                                depth, height, width)
    is_training       bool                             Whether in training mode.
    reduction_axes    list of dimensions (ints)        Axes to normalize over.
    momentum          0. <= float <= 1.                How quickly to update the
                                                       running statistics.
    beta              variable (shape is x shape with  Batch norm beta.
                      reduction_axes dims set to 1)
    gamma             variable (shape ")               Batch norm gamma.
    running_mean      constant (shape ")               Batch norm mean.
    running_variance  constant (shape ")               Batch norm variance.

Output:
    y                 variable (batch_size, channels,  The output.
                                shape...)
"""
batch_norm3d = Z.get('batch_norm3d', _my_batch_norm)
