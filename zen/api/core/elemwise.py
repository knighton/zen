from .. import backend as Z
from .data import clone


# Arithmetic.


abs = Z.abs
clip = Z.clip
neg = Z.neg
sign = Z.sign


# Cumulative.


def _my_cumprod(x, axis):
    prev_slices = [slice(None) for _ in range(x.ndim)]
    cur_slices = [slice(None) for _ in range(x.ndim)]
    out = clone(x)
    for i in range(x.shape[axis] - 1):
        prev_slices[axis] = slice(i, i + 1)
        cur_slices[axis] = slice(i + 1, i + 2)
        out[cur_slices] *= out[prev_slices]
    return out


def _my_cumsum(x, axis):
    prev_slices = [slice(None) for _ in range(x.ndim)]
    cur_slices = [slice(None) for _ in range(x.ndim)]
    out = clone(x)
    for i in range(x.shape[axis] - 1):
        prev_slices[axis] = slice(i, i + 1)
        cur_slices[axis] = slice(i + 1, i + 2)
        out[cur_slices] += out[prev_slices]
    return out


cumprod = Z.get('cumprod', _my_cumprod)
cumsum = Z.get('cumsum', _my_cumsum)

# Exponentiation.


def _my_expm1(x):
    return exp(x) - 1.


exp = Z.exp
expm1 = Z.get('expm1', _my_expm1)  # = exp(x) - 1


# Logarithms.


def _my_log10(x):
    return log(x) / log(10.)


def _my_log2(x):
    return log(x) / log(2.)


def _my_log1p(x):
    return log(1. + x)


log = Z.log
log10 = Z.get('log10', _my_log10)
log2 = Z.get('log2', _my_log2)
log1p = Z.get('log1p', _my_log1p)  # = log(1 + x)


# Power.


pow = Z.pow
sqrt = Z.sqrt
rsqrt = Z.rsqrt
square = Z.square


# Rounding.


ceil = Z.ceil    # Up.
floor = Z.floor  # Down.
round = Z.round  # To nearest.
trunc = Z.trunc  # Toward zero.


# Trigonometry.


def _my_arcsinh(x):
    return log(x + sqrt(square(x) + 1))


def _my_arccosh(x):
    return log(x + sqrt(square(x) - 1))


def _my_arctanh(x):
    return 0.5 * log((1. + x) / (1. - x))


sin = Z.sin
cos = Z.cos
tan = Z.tan
arcsin = Z.arcsin
arccos = Z.arccos
arctan = Z.arctan
sinh = Z.sinh
cosh = Z.cosh
tanh = Z.tanh
arcsinh = Z.get('arcsinh', _my_arcsinh)
arccosh = Z.get('arccosh', _my_arccosh)
arctanh = Z.get('arctanh', _my_arctanh)
