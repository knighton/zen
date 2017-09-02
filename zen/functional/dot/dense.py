def dense(x, kernel, bias):
    """
    x       tensor (batch_size, in_channels)
    kernel  tensor (out_channels, in_channels)
    bias    tensor (out_channels,)
    """
    return x.matmul(kernel) + bias
