from .. import backend as Z
from .data import clone


# Arithmetic.


abs = Z.abs
clip = Z.clip
neg = Z.neg
sign = Z.sign


# Cumulative.


def _cumprod(x, axis):
    prev_slices = [slice(None) for _ in range(x.ndim)]
    cur_slices = [slice(None) for _ in range(x.ndim)]
    out = clone(x)
    for i in range(x.shape[axis] - 1):
        prev_slices[axis] = slice(i, i + 1)
        cur_slices[axis] = slice(i + 1, i + 2)
        out[cur_slices] *= out[prev_slices]
    return out


def _cumsum(x, axis):
    prev_slices = [slice(None) for _ in range(x.ndim)]
    cur_slices = [slice(None) for _ in range(x.ndim)]
    out = clone(x)
    for i in range(x.shape[axis] - 1):
        prev_slices[axis] = slice(i, i + 1)
        cur_slices[axis] = slice(i + 1, i + 2)
        out[cur_slices] += out[prev_slices]
    return out


cumprod = Z.get('cumprod', _cumprod)
cumsum = Z.get('cumsum', _cumsum)


# Exponentiation.


def _expm1(x):
    return exp(x) - 1.


exp = Z.exp
expm1 = Z.get('expm1', _expm1)  # = exp(x) - 1


# Logarithms.


def _log10(x):
    return log(x) / log(10.)


def _log2(x):
    return log(x) / log(2.)


def _log1p(x):
    return log(1. + x)


log = Z.log
log10 = Z.get('log10', _log10)
log2 = Z.get('log2', _log2)
log1p = Z.get('log1p', _log1p)  # = log(1 + x)


# Power.


def _rsqrt(x):
    return 1. / sqrt(x)


def _sqrt(x):
    return pow(x, 0.5)


def _square(x):
    return pow(x, 2.)


pow = Z.pow
rsqrt = Z.get('rsqrt', _rsqrt)
sqrt = Z.get('sqrt', _sqrt)
square = Z.get('square', _square)


# Rounding.


ceil = Z.ceil    # Up.
floor = Z.floor  # Down.
round = Z.round  # To nearest.
trunc = Z.trunc  # Toward zero.


# Trigonometry.


def _sinh(x):
    return (exp(x) - exp(-x)) / 2.


def _cosh(x):
    return (exp(x) + exp(-x)) / 2.


def _tanh(x):
    e_x = exp(x)
    e_nx = exp(-x)
    return (e_x - e_nx) / (e_x + e_nx)


def _arcsinh(x):
    return log(x + sqrt(square(x) + 1.))


def _arccosh(x):
    return log(x + sqrt(square(x) - 1.))


def _arctanh(x):
    return 0.5 * log((1. + x) / (1. - x))


sin = Z.sin
cos = Z.cos
tan = Z.tan
arcsin = Z.arcsin
arccos = Z.arccos
arctan = Z.arctan
sinh = Z.get('sinh', _sinh)
cosh = Z.get('cosh', _cosh)
tanh = Z.get('tanh', _tanh)
arcsinh = Z.get('arcsinh', _arcsinh)
arccosh = Z.get('arccosh', _arccosh)
arctanh = Z.get('arctanh', _arctanh)
