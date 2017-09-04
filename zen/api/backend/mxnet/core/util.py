def to_shape(x, dim):
    if isinstance(x, int):
        assert 0 <= x
        return (x,) * dim
    elif isinstance(x, tuple):
        assert len(x) == dim
        return x
    else:
        assert False
