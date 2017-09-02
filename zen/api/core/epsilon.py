_EPSILON = 1e-6


def epsilon():
    return _EPSILON


def set_epsilon(x):
    assert isinstance(x, float), \
        'Epsilon must be a float (got a %s: %s).' % (type(x), x)
    assert 0. < x, 'Epsilon must be positive (got %s).' % x
    global _EPSILON
    _EPSILON = x
