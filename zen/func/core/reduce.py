from .. import engine as E
from .data import size
from .dtype import cast
from .elemwise import abs, ceil, clip, sqrt, square


def _any_all_common(x, axis=None, keepdims=False):
    x = abs(x)
    x = ceil(x)
    x = clip(x, 0, 1)
    x = cast(x, 'int64')
    x = sum(x, axis, keepdims)
    return clip(x, 0, 1)


def _all(x, axis=None, keepdims=False):
    x = _any_all_common(x, axis, keepdims)
    return sum(x) == size(x)


def _any(x, axis=None, keepdims=False):
    x = _any_all_common(x, axis, keepdims)
    return 0 < sum(x)


def _var(x, axis=None, keepdims=False):
    means = mean(x, axis, keepdims)
    x = square(abs(x - means))
    return mean(x, axis, keepdims)


def _std(x, axis=None, keepdims=False):
    return sqrt(_var(x, axis, keepdims))


all = E.get('all', _all)
any = E.get('any', _any)
argmax = E.argmax
argmin = E.argmin
max = E.max
mean = E.mean
min = E.min
prod = E.prod
std = E.get('std', _std)
sum = E.sum
var = E.get('var', _var)
