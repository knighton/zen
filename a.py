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


class Layer(object):
    def forward(self, x):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, w):
        self.w = w

    def forward(self, x):
        return x.mm(self.w)


class ReLU(Layer):
    def forward(self, x):
        return x.clamp(min=0)


tt = torch.cuda.FloatTensor
batch_size = 64
in_dim = 1000
hidden_dim = 100
num_classes = 10
learning_rate = 1e-6

x = constant(init(batch_size, in_dim), tt)
y_true = constant(init(batch_size, num_classes), tt)

w1 = variable(init(in_dim, hidden_dim), tt)
layer1 = Dense(w1)
layer2 = ReLU()
w2 = variable(init(hidden_dim, num_classes), tt)
layer3 = Dense(w2)

for t in range(500):
    y = layer1.forward(x)
    y = layer2.forward(y)
    y_pred = layer3.forward(y)

    loss = mean_squared_error(y_true, y_pred)
    print(t, loss.data[0])

    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()
