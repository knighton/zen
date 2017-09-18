import sys

from .accuracy import *  # noqa
from .core import *  # noqa
from .loss import *  # noqa
from .op import *  # noqa
from .optim import *  # noqa


def get(name, dim=None):
    if dim is not None:
        name = '%s%dd' % (name, dim)
    module = sys.modules[__name__]
    return getattr(module, name)
