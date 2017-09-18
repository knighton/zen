from ... import core as C


def global_avg_pool(x):
    """
    Global average pooling.

    Input:
                     1D   2D    3D
                     --   --    --
        x  variable  NCW  NCHW  NCDHW

    Output:
        y  variable  NC   NC    NC
    """
    ndim = C.get_ndim(x)
    assert ndim in {3, 4, 5}
    axes = list(range(ndim))[2:]
    return C.mean(x, axes)


global_avg_pool1d = global_avg_pool
global_avg_pool2d = global_avg_pool
global_avg_pool3d = global_avg_pool


def global_max_pool(x):
    """
    Global max pooling.

    Input:
                     1D   2D    3D
                     --   --    --
        x  variable  NCW  NCHW  NCDHW

    Output:
        y  variable  NC   NC    NC
    """
    ndim = C.get_ndim(x)
    assert ndim in {3, 4, 5}
    axes = list(range(ndim))[2:]
    return C.max(x, axes)


global_max_pool1d = global_max_pool
global_max_pool2d = global_max_pool
global_max_pool3d = global_max_pool
