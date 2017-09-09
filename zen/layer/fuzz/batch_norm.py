from ... import api as Z
from ... import init
from ..base import Transform, TransformSpec, Sugar


class BatchNormLayer(Transform):
    def __init__(self, ndim, momentum, beta, gamma, running_mean,
                 running_variance, reduction_axes):
        super().__init__()
        self.batch_norm = Z.get('batch_norm', ndim)
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


class BatchNormSpec(TransformSpec):
    def __init__(self, momentum=0.99, beta_init='zero', gamma_init='one',
                 running_mean_init='zero', running_variance_init='one',
                 ndim=None):
        super().__init__()
        self.momentum = momentum
        self.beta_init = init.get(beta_init)
        self.gamma_init = init.get(gamma_init)
        self.running_mean_init = init.get(running_mean_init)
        self.running_variance_init = init.get(running_variance_init)
        assert ndim in {None, 0, 1, 2, 3}
        self.ndim = ndim

    def build(self, in_shape, in_dtype):
        ndim = Z.verify_input_ndim(self.ndim, in_shape)
        reduction_axes = [0] + list(range(2, len(in_shape) + 1))
        shape = [1] + list(in_shape)
        for axis in reduction_axes:
            shape[axis] = 1
        shape = tuple(shape)
        beta = self.beta_init(shape, in_dtype)
        gamma = self.gamma_init(shape, in_dtype)
        running_mean = self.running_mean_init(shape, in_dtype)
        running_variance = self.running_variance_init(shape, in_dtype)
        layer = BatchNormLayer(ndim, self.momentum, beta, gamma, running_mean,
                               running_variance, reduction_axes)
        return layer, in_shape, in_dtype


BatchNorm = Sugar(BatchNormSpec)
BatchNorm0D = Sugar(BatchNormSpec, {'ndim': 0})
BatchNorm1D = Sugar(BatchNormSpec, {'ndim': 1})
BatchNorm2D = Sugar(BatchNormSpec, {'ndim': 2})
BatchNorm3D = Sugar(BatchNormSpec, {'ndim': 3})
