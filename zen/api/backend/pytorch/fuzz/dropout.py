from torch.nn import functional as F


def dropout(x, is_training, rate):
    return F.dropout(x, rate, is_training, False)


dropout0d = dropout
dropout1d = dropout
dropout2d = dropout
dropout3d = dropout
