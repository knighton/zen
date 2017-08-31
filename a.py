import torch

from zen import constant
from zen.layer import *


def mean_squared_error(true, pred):
    return (true - pred).pow(2).sum()


batch_size = 64
in_dim = 1000
hidden_dim = 100
num_classes = 10
learning_rate = 1e-6

x = constant(init(batch_size, in_dim))
y_true = constant(init(batch_size, num_classes))

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
