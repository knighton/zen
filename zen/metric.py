import sys

from .functional.accuracy import *  # noqa
from .functional.loss import *  # noqa


def get(x):
    if callable(x):
        return x
    elif isinstance(x, str):
        module = sys.modules[__name__]
        return getattr(module, x)
    else:
        assert False
