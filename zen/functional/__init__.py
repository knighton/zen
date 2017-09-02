import sys

from .accuracy import *
from .core import *
from .curve import *
from .dot import *
from .fuzz import *
from .loss import *
from .shape import *


def get(name, dim=None):
    if dim is not None:
        name = '%s%dd' % (name, dim)
    module = sys.modules[__name__]
    return getattr(module, name)
