import numpy as np
from scipy.stats import truncnorm
import sys

from . import func as Z


class Initializer(object):
    def __call__(self, shape, dtype=None):
        raise NotImplementedError


class Orthogonal(Initializer):
    def __init__(self, gain=1.):
        self.gain = gain

    def __call__(self, shape, dtype=None):
        dtype = dtype or Z.floatx()
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_rows, num_cols)
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        arr = self.gain * q[:shape[0], :shape[1]]
        return arr.astype(dtype)


orthogonal = Orthogonal


class Eye(Initializer):
    def __init__(self, gain=1.):
        self.gain = gain

    def __call__(self, shape, dtype=None):
        assert len(shape) == 2
        assert shape[0] == shape[1]
        dtype = dtype or Z.floatx()
        arr = self.gain * np.eye(shape[0])
        return arr.astype(dtype)


eye = Eye


class Zero(Initializer):
    def __call__(self, shape, dtype=None):
        dtype = dtype or Z.floatx()
        return np.zeros(shape).astype(dtype)


zero = Zero


class One(Initializer):
    def __call__(self, shape, dtype=None):
        dtype = dtype or Z.floatx()
        return np.ones(shape).astype(dtype)


one = One


class Full(Initializer):
    def __init__(self, value=0.):
        self.value = value

    def __call__(self, shape, dtype=None):
        dtype = dtype or Z.floatx()
        return np.full(shape, self.value).astype(dtype)


full = Full


def _normal(mean, std, shape, dtype):
    return np.random.normal(0., std, shape).astype(dtype)


class Normal(Initializer):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, shape, dtype=None):
        dtype = dtype or Z.floatx()
        return _normal(self.mean, self.std, shape, dtype)


normal = Normal


def _truncated_normal(mean, std, min_stds, max_stds, shape, dtype):
    arr = truncnorm(min_stds, max_stds)
    arr = arr.rvs(np.prod(shape)).reshape(shape)
    arr = arr * std + mean
    return arr.astype(dtype)


class TruncatedNormal(Initializer):
    def __init__(self, mean=0., std=1., min_stds=-2., max_stds=2.):
        self.mean = mean
        self.std = std
        self.min_stds = min_stds
        self.max_stds = max_stds

    def __call__(self, shape, dtype=None):
        dtype = dtype or Z.floatx()
        return _truncated_normal(self.mean, self.std, self.min_stds,
                                 self.max_stds, shape, dtype)


truncated_normal = TruncatedNormal


def _uniform(min_, max_, shape, dtype):
    return np.random.uniform(min_, max_, shape).astype(dtype)


class Uniform(Initializer):
    def __init__(self, min_=-0.05, max_=0.05):
        self.min = min_
        self.max = max_

    def __call__(self, shape, dtype=None):
        dtype = dtype or Z.floatx()
        return _uniform(self.min, self.max, shape, dtype)


uniform = Uniform


def _compute_fans(shape):
    z = len(shape)
    if z == 1:
        fan_in = fan_out = shape[0]
    elif z == 2:
        fan_in = shape[1]
        fan_out = shape[0]
    elif z in {3, 4, 5}:
        both = np.prod(shape[2:])
        fan_in = shape[1] * both
        fan_out = shape[0] * both
    else:
        fan_in = fan_out = np.sqrt(np.prod(shape[0]))
    return fan_in, fan_out


def _compute_scale(mode, fan_in, fan_out):
    if mode == 'fan_in':
        return fan_in
    elif mode == 'fan_avg':
        return (fan_in + fan_out) / 2.
    elif mode == 'fan_out':
        return fan_out
    else:
        assert False


def _compute_distribution(dist, scale, shape, dtype):
    if dist == 'normal':
        std = np.sqrt(scale)
        return _normal(0., std, shape, dtype)
    elif dist == 'truncated_normal':
        std = np.sqrt(scale)
        return _truncated_normal(0., std, -2., 2., shape, dtype)
    elif dist == 'uniform':
        limit = np.sqrt(3. * scale)
        return _uniform(-limit, limit, shape, dtype)
    else:
        assert False


class ScaledDistribution(Initializer):
    def __init__(self, scale=1., mode='fan_in', dist='truncated_normal'):
        self.scale = scale
        assert mode in {'fan_in', 'fan_avg', 'fan_out'}
        self.mode = mode
        assert dist in {'normal', 'truncated_normal', 'uniform'}
        self.dist = dist

    def __call__(self, shape, dtype=None):
        dtype = dtype or Z.floatx()
        fan_in, fan_out = _compute_fans(shape)
        scale = self.scale / _compute_scale(self.mode, fan_in, fan_out)
        return _compute_distribution(self.dist, scale, shape, dtype)


glorot_normal = lambda: ScaledDistribution(1., 'fan_avg', 'truncated_normal')
glorot_uniform = lambda: ScaledDistribution(1., 'fan_avg', 'uniform')
he_normal = lambda: ScaledDistribution(2., 'fan_in', 'truncated_normal')
he_uniform = lambda: ScaledDistribution(2., 'fan_in', 'uniform')
lecun_normal = lambda: ScaledDistribution(1., 'fan_in', 'truncated_normal')
lecun_uniform = lambda: ScaledDistribution(1., 'fan_in', 'uniform')


def get(x):
    if isinstance(x, Initializer):
        return x
    elif isinstance(x, str):
        module = sys.modules[__name__]
        return getattr(module, x)()
    else:
        assert False
