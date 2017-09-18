from torch.nn import functional as F


def batch_norm(x, is_training, reduction_axes, momentum, beta, gamma,
               running_mean, running_variance):
    running_mean = running_mean.squeeze().data
    running_variance = running_variance.squeeze().data
    gamma = gamma.squeeze()
    beta = beta.squeeze()
    return F.batch_norm(x, running_mean, running_variance, gamma, beta,
                        is_training, momentum, 1e-3)


batch_norm0d = batch_norm
batch_norm1d = batch_norm
batch_norm2d = batch_norm
batch_norm3d = batch_norm
