import sys

from .hook import Hook
from .stop import *     # noqa
from .verbose import *  # noqa


def get(x):
    if isinstance(x, Hook):
        return x
    elif isinstance(x, str):
        module = sys.modules[__name__]
        return getattr(module, x)()
    else:
        assert False
