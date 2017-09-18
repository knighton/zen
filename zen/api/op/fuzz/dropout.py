from ... import backend as Z


"""
Dropout.  For spatial dropout, see `spatial_dropout`.

    x, is_training, rate -> y

Input:
                           0D  1D   2D    3D
                           --  --   --    --
    x            variable  NC  NCW  NCHW  NCDHW
    is_training  bool
    rate         0 to 1

Output:
    y            variable  NC  NCW  NCHW  NCDHW
"""
dropout = Z.dropout
dropout0d = Z.dropout0d
dropout1d = Z.dropout1d
dropout2d = Z.dropout2d
dropout3d = Z.dropout3d
