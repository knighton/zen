import sys

from .functional import binary_accuracy, binary_cross_entropy, \
                        categorical_accuracy, categorical_cross_entropy, \
                        mean_squared_error


def get(x):
    if callable(x):
        return x
    elif isinstance(x, str):
        module = sys.modules[__name__]
        return getattr(module, x)
    else:
        assert False
