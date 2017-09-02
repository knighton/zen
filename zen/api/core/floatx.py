def floatx():
    return _FLOATX


def set_floatx(dtype):
    assert 'float' in dtype
    global _FLOATX
    _FLOATX = dtype


set_floatx('float32')
