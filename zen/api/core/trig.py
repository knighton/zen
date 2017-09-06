from .. import backend as Z
from .elemwise import log, square, sqrt


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
