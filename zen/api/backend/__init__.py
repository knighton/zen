import os
import sys


backend = os.environ.get('ZEN_BACKEND', 'pytorch')
if backend == 'mxnet':
    from .mxnet import *  # noqa
elif backend == 'pytorch':
    from .pytorch import *  # noqa
else:
    assert False, 'Unsupported backend: %s.' % backend


def get(name, fallback=None):
    module = sys.modules[__name__]
    ret = getattr(module, name, fallback)
    assert ret
    return ret
