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


tt = torch.cuda.FloatTensor

batch_size, in_dim, hidden_dim, num_classes = 64, 1000, 100, 10

x = constant(init(batch_size, in_dim), tt)
y_true = constant(init(batch_size, num_classes), tt)

w1 = variable(init(in_dim, hidden_dim), tt)
w2 = variable(init(hidden_dim, num_classes), tt)

learning_rate = 1e-6
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    loss = mean_squared_error(y_true, y_pred)
    print(t, loss.data[0])

    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()
