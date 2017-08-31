import sys

from .mvo import MVO
from .optimizer import Optimizer


mvo = MVO


def get(x):
    if isinstance(x, Optimizer):
        return x
    elif isinstance(x, str):
        module = sys.modules[__name__]
        return getattr(module, x)()
    else:
        assert False
