import sys

from .accuracy import *  # noqa
from .core import *  # noqa
from .curve import *  # noqa
from .dot import *  # noqa
from .fuzz import *  # noqa
from .loss import *  # noqa
from .proj import *  # noqa
from .shape import *  # noqa


def get(name, dim=None):
    if dim is not None:
        name = '%s%dd' % (name, dim)
    module = sys.modules[__name__]
    return getattr(module, name)
