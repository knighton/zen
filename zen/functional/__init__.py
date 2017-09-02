import sys
import torch
from torch.nn import functional as F

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


def floatx():
    return 'float32'


def check_dim(dim):
    assert isinstance(dim, int), \
        'Dimension must be an integer (got %s): %s.' % (type(dim), dim)
    assert 1 <= dim, 'Dimension must be positive: %d.' % dim


def reshape(x, out_shape):
    """
    x          tensor (batch_size, in_shape...)  Data to reshape.
    out_shape  shape...                          Shape without batch_size dim.
    """
    batch_out_shape = (x.size(0),) + out_shape
    return x.view(batch_out_shape)
