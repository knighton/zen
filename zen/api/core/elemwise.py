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

# Exponentiation.
exp = Z.exp
expm1 = Z.get('expm1', _my_expm1)  # = exp(x) - 1

# Logarithms.
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
sin = Z.sin
cos = Z.cos
tan = Z.tan
arcsin = Z.arcsin
arccos = Z.arccos
arctan = Z.arctan
sinh = Z.sinh
cosh = Z.cosh
tanh = Z.tanh
_my_arcsinh = lambda x: log(x + sqrt(square(x) + 1))
_my_arccosh = lambda x: log(x + sqrt(square(x) - 1))
_my_arctanh = lambda x: 0.5 * log((1. + x) / (1. - x))
arcsinh = Z.get('arcsinh', _my_arcsinh)
arccosh = Z.get('arccosh', _my_arccosh)
arctanh = Z.get('arctanh', _my_arctanh)
