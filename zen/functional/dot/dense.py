from torch.nn import functional as F


def dense(x, kernel, bias):
    """
    x       tensor (batch_size, in_channels)
    kernel  tensor (out_channels, in_channels)
    bias    tensor (out_channels,)
    """
    return F.linear(x, kernel, bias)
