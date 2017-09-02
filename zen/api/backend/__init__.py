import sys

from .pytorch import *  # noqa


def get(name, fallback=None):
    module = sys.modules[__name__]
    ret = getattr(module, name, fallback)
    assert ret
    return ret
