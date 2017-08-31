from argparse import ArgumentParser

from zen.dataset.mnist import load_mnist
from zen.layer import *
from zen.optim import MVO
from zen.transform.one_hot import one_hot


(x_train, y_train), (x_val, y_val) = load_mnist()
image_shape = x_train.shape[1:]
num_classes = y_train.max() + 1
y_train = one_hot(y_train, num_classes)
y_val = one_hot(y_val, num_classes)
data = (x_train, y_train), (x_val, y_val)

spec = SequenceSpec(
    Input(image_shape),
    Flatten,
    Dense(256),
    ReLU,
    Dense(64),
    ReLU,
    Dense(num_classes),
    Softmax
)

model, out_shape, out_dtype = spec.build()

opt = MVO(0.2)

batch_size = 64
num_epochs = 1000
model.train_classifier(data, opt, batch_size, num_epochs)
