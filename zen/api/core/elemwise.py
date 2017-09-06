from .. import backend as Z


def _my_expm1(x):
    return exp(x) - 1.


def _my_log10(x):
    return log(x) / log(10.)


def _my_log2(x):
    return log(x) / log(2.)


def _my_log1p(x):
    return log(1. + x)


# Arithmetic.
abs = Z.abs
clip = Z.clip
neg = Z.neg
sign = Z.sign

# Cumulative.
cumprod = Z.get('cumprod', None)  # TODO: implement fallback.
cumsum = Z.get('cumsum', None)    # TODO: implement fallback.

# Exponent.
exp = Z.exp
expm1 = Z.get('expm1', _my_expm1)  # = exp(x) - 1

# Logarithm.
log = Z.log
log10 = Z.get('log10', _my_log10)
log2 = Z.get('log2', _my_log2)
log1p = Z.get('log1p', _my_log1p)  # = log(1 + x)

# Power.
pow = Z.pow
sqrt = Z.sqrt
rsqrt = Z.rsqrt
square = Z.square
