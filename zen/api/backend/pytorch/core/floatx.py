_POSSIBLE_FLOATX = {'float16', 'float32', 'float64'}
_FLOATX = 'float32'


def floatx():
    return _FLOATX


def set_floatx(dtype):
    assert isinstance(dtype, str), \
        ('Floatx must be a string dtype (like \'float32\') (got a %s: %s).' %
         (dtype.__class__.__name__, dtype))
    assert dtype in _POSSIBLE_FLOATX, \
        'Float must be one of %s (got %s).' % (_POSSIBLE_FLOATX, dtype)
    global _FLOATX
    _FLOATX = dtype


set_floatx('float32')
