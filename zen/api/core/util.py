def check_dim(dim):
    assert isinstance(dim, int), \
        'Dimension must be an integer (got %s): %s.' % (type(dim), dim)
    assert 1 <= dim, 'Dimension must be positive: %d.' % dim


def to_shape(x, dim):
    if isinstance(x, int):
        return (x,) * dim
    elif isinstance(x, tuple):
        assert len(x) == dim
        return x
    else:
        assert False


def check_out_channels(out_channels, name):
    assert out_channels is None or _is_dim(out_channels), \
        ('The output channels (`%s`) must either be a dimension (int) or ' +
         'None to match the input channels.  We got a %s: %s.') % \
        (name, _type(out_channels), out_channels)


def check_input_ndim(ndim, possible_dims, name):
    assert ndim is None or ndim in possible_dims, \
        ('The data dimensionality (`%s`) must be either in %s or None to ' +
         'infer from input (got a %s: %s).') % \
        (name, possible_dims, _type(ndim), ndim)


def _type(x):
    return x.__class__.__name__


def _is_dim(dim):
    return isinstance(dim, int) and 0 < dim


def _is_shape(shape):
    if not isinstance(shape, tuple):
        return False
    for dim in shape:
        if not _is_dim(dim):
            return False
    return True


def _is_dim_or_shape_error(arg, ndim, name):
    if ndim is None:
        msg = ('`%s` must be either a dimension (1 <= int) or a shape ' +
               '(tuple of such ints).  We got a %s: %s.') % \
              (name, _type(arg), arg)
    else:
        msg = ('`%s` must be either a dimension (1 <= int) or a %d-shape ' +
               '(%d-tuple of such ints).  We got a %s: %s.') % \
              (name, ndim, ndim, _type(arg), arg)
    return msg


def check_dim_or_shape(arg, ndim, name):
    if isinstance(arg, int):
        assert _is_dim(arg), _is_dim_or_shape_error(arg, ndim, name)
    elif hasattr(arg, '__getitem__'):
        assert _is_shape(arg), _is_dim_or_shape_error(arg, ndim, name)
    else:
        assert False, _is_dim_or_shape_error(arg, ndim, name)


def _is_coord(coord):
    return isinstance(coord, int) and 0 <= coord


def _is_coords(coords):
    if not isinstance(coords, tuple):
        return False
    for coord in coords:
        if not _is_coord(coord):
            return False
    return True


def _check_coord_or_coords_error(arg, ndim, name):
    if ndim is None:
        msg = ('`%s` must be either a coord (0 <= int) or coords (tuple of ' +
               'such ints).  We got a(n) %s: %s.') % (name, _type(arg), arg)
    else:
        msg = ('`%s` must be either a coord (0 <= int) or %d coords ' +
               '(%d-tuple of such ints).  We got a(n) %s: %s.') % \
              (name, ndim, ndim, _type(arg), arg)
    return msg


def check_coord_or_coords(arg, ndim, name):
    err_msg = lambda: _check_coord_or_coords_error(arg, ndim, name)
    if isinstance(arg, int):
        assert _is_coord(arg), err_msg()
    elif hasattr(arg, '__getitem__'):
        assert _is_coords(arg), err_msg()
    else:
        assert False, err_msg()


def verify_input_ndim(required_ndim, in_shape):
    if required_ndim is None:
        return len(in_shape) - 1
    assert _is_shape(in_shape, required_ndim + 1), \
        ('The layer\'s expected number of input dimensions (%s) does not ' +
         'match the actual number of input dimensions passed to build(): ' +
         '%s.') % (required_ndim, in_shape)
    return required_ndim
