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

    def forward(self, x):
        raise NotImplementedError


class Layer(Layerlike):
    def __init__(self):
        self._params = []

    def add_param(self, arr):
        param = variable(arr, tt)
        self._params.append(param)
        return param

    def get_params(self):
        return self._params


class Spec(object):
    def build(self):
        raise NotImplementedError


class DenseLayer(Layer):
    def __init__(self, w, b):
        super().__init__()
        self.w = self.add_param(w)
        self.b = self.add_param(b)

    def forward(self, x):
        return x.mm(self.w) + self.b


class DenseSpec(Spec):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

    def build(self):
        return DenseLayer(init(self.in_dim, self.out_dim), init(self.out_dim))


class ReLULayer(Layer):
    def get_params(self):
        return []

    def forward(self, x):
        return x.clamp(min=0)


class ReLUSpec(Spec):
    def build(self):
        return ReLULayer()


class SequenceLayer(Layerlike):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


class SequenceSpec(Spec):
    def __init__(self, *specs):
        self.specs = specs

    def build(self):
        layers = []
        for spec in self.specs:
            layer = spec.build()
            layers.append(layer)
        return SequenceLayer(layers)


tt = torch.cuda.FloatTensor
batch_size = 64
in_dim = 1000
hidden_dim = 100
num_classes = 10
learning_rate = 1e-6

x = constant(init(batch_size, in_dim), tt)
y_true = constant(init(batch_size, num_classes), tt)

spec = SequenceSpec(
    DenseSpec(in_dim, hidden_dim),
    ReLUSpec(),
    DenseSpec(hidden_dim, num_classes)
)
model = spec.build()
params = model.get_params()

for t in range(500):
    y_pred = model.forward(x)

    loss = mean_squared_error(y_true, y_pred)
    print(t, loss.data[0])

    loss.backward()

    for param in params:
        param.data -= learning_rate * param.grad.data
        param.grad.data.zero_()
