from .. import engine as E
from .data import clone


# Arithmetic.


abs = E.abs
clip = E.clip
neg = E.neg
sign = E.sign


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


cumprod = E.get('cumprod', _cumprod)
cumsum = E.get('cumsum', _cumsum)


# Exponentiation.


def _expm1(x):
    return exp(x) - 1.


exp = E.exp
expm1 = E.get('expm1', _expm1)  # = exp(x) - 1


# Logarithms.


def _log10(x):
    return log(x) / log(10.)


def _log2(x):
    return log(x) / log(2.)


def _log1p(x):
    return log(1. + x)


log = E.log
log10 = E.get('log10', _log10)
log2 = E.get('log2', _log2)
log1p = E.get('log1p', _log1p)  # = log(1 + x)


# Power.


def _rsqrt(x):
    return 1. / sqrt(x)


def _sqrt(x):
    return pow(x, 0.5)


def _square(x):
    return pow(x, 2.)


pow = E.pow
rsqrt = E.get('rsqrt', _rsqrt)
sqrt = E.get('sqrt', _sqrt)
square = E.get('square', _square)


# Rounding.


ceil = E.ceil    # Up.
floor = E.floor  # Down.
round = E.round  # To nearest.
trunc = E.trunc  # Toward zero.


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


sin = E.sin
cos = E.cos
tan = E.tan
arcsin = E.arcsin
arccos = E.arccos
arctan = E.arctan
sinh = E.get('sinh', _sinh)
cosh = E.get('cosh', _cosh)
tanh = E.get('tanh', _tanh)
arcsinh = E.get('arcsinh', _arcsinh)
arccosh = E.get('arccosh', _arccosh)
arctanh = E.get('arctanh', _arctanh)
