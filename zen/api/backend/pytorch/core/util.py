def to_one(x):
    if isinstance(x, int):
        return x
    elif isinstance(x, tuple):
        assert len(x) == 1
        return x[0]
    else:
        assert False
