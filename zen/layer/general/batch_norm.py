from ... import functional as F
from ... import init
from ..layer import Layer, Spec, Sugar


class BatchNormLayer(Layer):
    def __init__(self, dim, momentum, beta, gamma, running_mean,
                 running_variance, reduction_axes):
        super().__init__()
        self.batch_norm = F.get('batch_norm', dim)
        self.momentum = momentum
        self.beta = self.add_param(beta)
        self.gamma = self.add_param(gamma)
        self.running_mean = self.add_param(running_mean, trainable=False)
        self.running_variance = self.add_param(running_variance,
                                               trainable=False)
        self.reduction_axes = reduction_axes

    def forward(self, x, is_training):
        return self.batch_norm(x, is_training, self.reduction_axes,
                               self.momentum, self.beta, self.gamma,
                               self.running_mean, self.running_variance)


class BatchNormSpec(Spec):
    def __init__(self, momentum=0.99, beta_init='zero', gamma_init='one',
                 running_mean_init='zero', running_variance_init='one',
                 dim=None):
        super().__init__()
        self.momentum = momentum
        self.beta_init = init.get(beta_init)
        self.gamma_init = init.get(gamma_init)
        self.running_mean_init = init.get(running_mean_init)
        self.running_variance_init = init.get(running_variance_init)
        assert dim in {None, 0, 1, 2, 3}
        self.dim = dim

    def build(self, in_shape, in_dtype):
        if self.dim is None:
            dim = len(in_shape) - 1
        else:
            assert len(in_shape) == self.dim + 1
            dim = self.dim
        reduction_axes = [0] + list(range(2, len(in_shape) + 1))
        shape = [1] * (len(in_shape) + 1)
        shape[1] = in_shape[0]
        beta = self.beta_init(shape, in_dtype)
        gamma = self.gamma_init(shape, in_dtype)
        running_mean = self.running_mean_init(shape, in_dtype)
        running_variance = self.running_variance_init(shape, in_dtype)
        layer = BatchNormLayer(dim, self.momentum, beta, gamma, running_mean,
                               running_variance, reduction_axes)
        return layer, in_shape, in_dtype


BatchNorm = Sugar(BatchNormSpec)
BatchNorm0D = Sugar(BatchNormSpec, {'dim': 0})
BatchNorm1D = Sugar(BatchNormSpec, {'dim': 1})
BatchNorm2D = Sugar(BatchNormSpec, {'dim': 2})
BatchNorm3D = Sugar(BatchNormSpec, {'dim': 3})
