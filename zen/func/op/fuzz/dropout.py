from ... import engine as E


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
dropout = E.dropout
dropout0d = E.dropout0d
dropout1d = E.dropout1d
dropout2d = E.dropout2d
dropout3d = E.dropout3d
