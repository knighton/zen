def reshape(x, out_shape):
    batch_out_shape = (x.size(0),) + out_shape
    return x.view(batch_out_shape)
