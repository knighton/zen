def check_dim(dim):
    assert isinstance(dim, int), \
        'Dimension must be an integer (got %s): %s.' % (type(dim), dim)
    assert 1 <= dim, 'Dimension must be positive: %d.' % dim


def to_one(x):
    if isinstance(x, int):
        return x
    elif isinstance(x, tuple):
        assert len(x) == 1
        return x[0]
    else:
        assert False


def to_shape(x, dim):
    if isinstance(x, int):
        return (x,) * dim
    elif isinstance(x, tuple):
        assert len(x) == dim
        return x
    else:
        assert False
