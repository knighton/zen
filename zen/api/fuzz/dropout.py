from .. import backend as Z


"""
ND dropout.  For spatial dropout, see `spatial_dropout`.

Input:
    x            variable (batch_size, channels,  The input.
                           shape...)
    is_training  bool                             Whether to drop.
    rate         0. < float < 1.                  Fraction dropped.

Output:
    y            variable (batch_size, channels,  The output.
                           shape...)
"""
dropout = Z.dropout


"""
0D dropout.

Input:
    x            variable (batch_size, channels)  The input.
    is_training  bool                             Whether to drop.
    rate         0. < float < 1.                  Fraction dropped.

Output:
    y            variable (batch_size, channels)  The output.
"""
dropout0d = Z.dropout0d


"""
1D dropout.  For spatial dropout, see `spatial_dropout`.

Input:
    x            variable (batch_size, channels,  The input.
                           width)
    is_training  bool                             Whether to drop.
    rate         0. < float < 1.                  Fraction dropped.

Output:
    y            variable (batch_size, channels,  The output.
                           width)
"""
dropout1d = Z.dropout1d


"""
2D dropout.  For spatial dropout, see `spatial_dropout`.

Input:
    x            variable (batch_size, channels,  The input.
                           height, width)
    is_training  bool                             Whether to drop.
    rate         0. < float < 1.                  Fraction dropped.

Output:
    y            variable (batch_size, channels,  The output.
                           height, width)
"""
dropout2d = Z.dropout2d


"""
3D dropout.  For spatial dropout, see `spatial_dropout`.

Input:
    x            variable (batch_size, channels,  The input.
                           depth, height, width)
    is_training  bool                             Whether to drop.
    rate         0. < float < 1.                  Fraction dropped.

Output:
    y            variable (batch_size, channels,  The output.
                           depth, height, width)
"""
dropout3d = Z.dropout3d
