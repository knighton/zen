from torch.nn import functional as F


def spatial_dropout2d(x, is_training, rate):
    return F.dropout2d(x, rate, is_training, False)


def spatial_dropout3d(x, is_training, rate):
    return F.dropout3d(x, rate, is_training, False)
