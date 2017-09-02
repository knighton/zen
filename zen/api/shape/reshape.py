def reshape(x, out_shape):
    """
    x          tensor (batch_size, in_shape...)  Data to reshape.
    out_shape  shape...                          Shape without batch_size dim.
    """
    batch_out_shape = (x.size(0),) + out_shape
    return x.view(batch_out_shape)
