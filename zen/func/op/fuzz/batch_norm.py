from ... import core as C
from ... import engine as E


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
    x_running[:] = momentum * x_running + (1. - momentum) * x_new


def _batch_norm(x, is_training, reduction_axes, momentum, beta, gamma,
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
Batch normalization.

    (x, is_training, reduction_axes, momentum, beta, gamma, running_mean,
     running_variance) -> y

Input:
                                    0D  1D   2D    3D
                                    --  --   --    --
    x                 variable      NC  NCW  NCHW  NCDHW
    is_training       bool
    reduction_axes    list of ints
    momentum          0 to 1
    beta              variable      1C  1C1  1C11  1C111
    gamma             variable      1C  1C1  1C11  1C111
    running_mean      constant      1C  1C1  1C11  1C111
    running_variance  constant      1C  1C1  1C11  1C111

Output:
    y                 variable      NC  NCW  NCHW  NCDHW

The variables beta, gamma, running_mean, and running_variance are the same shape
as `x` but with a length of 1 on every reduced axis.  Reduction axes is going to
be everything but the channels axis.
"""
batch_norm = E.get('batch_norm', _batch_norm)
batch_norm0d = E.get('batch_norm0d', _batch_norm)
batch_norm1d = E.get('batch_norm1d', _batch_norm)
batch_norm2d = E.get('batch_norm2d', _batch_norm)
batch_norm3d = E.get('batch_norm3d', _batch_norm)
