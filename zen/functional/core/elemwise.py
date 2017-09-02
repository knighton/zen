import numpy as np


def clip(x, min_value=-np.inf, max_value=np.inf):
    return x.clamp(min_value, max_value)


def sqrt(x):
    return x.sqrt()


def square(x):
    return x.pow(2)
