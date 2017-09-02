from .. import backend as Z


"""
ND dropout.  For spatial dropout, see `spatial_dropout`.

Input:
    x            variable (NC...)  Input data.
    is_training  bool              Whether to drop.
    rate         0. < float < 1.   Fraction dropped.

Output:
    y            variable (NC...)  Output data.
"""
dropout = Z.dropout


"""
0D dropout (vectors).

Input:
    x            variable (NC)     Input data.
    is_training  bool              Whether to drop.
    rate         0. < float < 1.   Fraction dropped.

Output:
    y            variable (NC)     Output data.
"""
dropout0d = Z.dropout0d


"""
1D dropout (sequences).  For spatial dropout, see `spatial_dropout`.

Input:
    x            variable (NCW)    Input data.
    is_training  bool              Whether to drop.
    rate         0. < float < 1.   Fraction dropped.

Output:
    y            variable (NCW)    Output data.
"""
dropout1d = Z.dropout1d


"""
2D dropout (images).  For spatial dropout, see `spatial_dropout`.

Input:
    x            variable (NCHW)   Input data.
    is_training  bool              Whether to drop.
    rate         0. < float < 1.   Fraction dropped.

Output:
    y            variable (NCHW)   Output data.
"""
dropout2d = Z.dropout2d


"""
3D dropout (video).  For spatial dropout, see `spatial_dropout`.

Input:
    x            variable (NCDHW)  Input data.
    is_training  bool              Whether to drop.
    rate         0. < float < 1.   Fraction dropped.

Output:
    y            variable (NCDHW)  Output data.
"""
dropout3d = Z.dropout3d
