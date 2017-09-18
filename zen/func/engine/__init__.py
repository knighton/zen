import os
import sys


engine = os.environ.get('ZEN_BACKEND', 'pytorch')
if engine == 'mxnet':
    from .mxnet import *  # noqa
elif engine == 'pytorch':
    from .pytorch import *  # noqa
else:
    assert False, 'Unsupported engine: %s.' % engine


def get(name, fallback=None):
    module = sys.modules[__name__]
    ret = getattr(module, name, fallback)
    assert ret
    return ret
