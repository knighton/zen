import numpy as np
import torch
from torch.autograd import Variable


def variable(arr, tensor_type):
    tensor = torch.from_numpy(arr).type(tensor_type)
    return Variable(tensor, requires_grad=True)


def constant(arr, tensor_type):
    tensor = torch.from_numpy(arr).type(tensor_type)
    return Variable(tensor, requires_grad=False)


def init(*shape):
    return np.random.normal(0., 1., shape)


def mean_squared_error(true, pred):
    return (true - pred).pow(2).sum()


class Layerlike(object):
    def get_params(self):
        raise NotImplementedError

    def forward(self, x, is_training):
        raise NotImplementedError


class Layer(Layerlike):
    def __init__(self):
        super().__init__()
        self._params = []

    def add_param(self, arr):
        param = variable(arr, tt)
        self._params.append(param)
        return param

    def get_params(self):
        return self._params


class Spec(object):
    def build(self, in_shape=None, in_dtype=None):
        raise NotImplementedError


class InputLayer(Layer):
    def forward(self, x, is_training):
        return x


class InputSpec(Spec):
    def __init__(self, shape, dtype):
        super().__init__()
        self.shape = shape
        self.dtype = dtype

    def build(self, in_shape=None, in_dtype=None):
        if in_shape is not None:
            assert in_shape == self.shape
        if in_dtype is not None:
            assert in_dtype == self.dtype
        return InputLayer(), self.shape, self.dtype


class DenseLayer(Layer):
    def __init__(self, w, b):
        super().__init__()
        self.w = self.add_param(w)
        self.b = self.add_param(b)

    def forward(self, x, is_training):
        return x.mm(self.w) + self.b


class DenseSpec(Spec):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim

    def build(self, in_shape=None, in_dtype=None):
        in_dim, = in_shape
        out_shape = self.out_dim,
        layer = DenseLayer(init(in_dim, self.out_dim), init(self.out_dim))
        return layer, out_shape, in_dtype


class ReLULayer(Layer):
    def get_params(self):
        return []

    def forward(self, x, is_training):
        return x.clamp(min=0)


class ReLUSpec(Spec):
    def build(self, in_shape=None, in_dtype=None):
        return ReLULayer(), in_shape, in_dtype


class SequenceLayer(Layerlike):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params

    def forward(self, x, is_training):
        for layer in self.layers:
            x = layer.forward(x, is_training)
        return x


class SequenceSpec(Spec):
    def __init__(self, *specs):
        super().__init__()
        self.specs = specs

    def build(self, in_shape=None, in_dtype=None):
        layers = []
        for spec in self.specs:
            layer, in_shape, in_dtype = spec.build(in_shape, in_dtype)
            layers.append(layer)
        return SequenceLayer(layers), in_shape, in_dtype


tt = torch.cuda.FloatTensor
batch_size = 64
in_dim = 1000
hidden_dim = 100
num_classes = 10
learning_rate = 1e-6

x = constant(init(batch_size, in_dim), tt)
y_true = constant(init(batch_size, num_classes), tt)

spec = SequenceSpec(
    InputSpec((in_dim,), 'float32'),
    DenseSpec(hidden_dim),
    ReLUSpec(),
    DenseSpec(num_classes)
)
model, out_shape, out_dtype = spec.build()
params = model.get_params()

for t in range(500):
    y_pred = model.forward(x, True)

    loss = mean_squared_error(y_true, y_pred)
    print(t, loss.data[0])

    loss.backward()

    for param in params:
        param.data -= learning_rate * param.grad.data
        param.grad.data.zero_()
