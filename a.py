from argparse import ArgumentParser

from zen import constant
from zen.dataset.mnist import load_mnist
from zen.functional import categorical_accuracy, categorical_cross_entropy
from zen.layer import *
from zen.optim import MVO
from zen.transform.one_hot import one_hot


(x_train, y_train), (x_val, y_val) = load_mnist()
image_shape = x_train.shape[1:]
num_classes = y_train.max() + 1
y_train = one_hot(y_train, num_classes)
y_val = one_hot(y_val, num_classes)

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
opt.set_params(model.get_params())

batch_size = 64
num_epochs = 1000
num_train_batches = len(x_train) // batch_size
num_val_batches = len(x_val) // batch_size
for i in range(num_epochs):
    train_losses = []
    train_accs = []
    for j in range(num_train_batches):
        a = j * batch_size
        z = (j + 1) * batch_size
        x = constant(x_train[a:z])
        y_true = constant(y_train[a:z])
        y_pred = model.forward(x, True)
        loss = categorical_cross_entropy(y_true, y_pred)
        train_losses.append(loss.data.cpu().numpy()[0])
        loss.backward()
        acc = categorical_accuracy(y_true, y_pred)
        train_accs.append(acc.data.cpu().numpy()[0])
        opt.step()
    val_losses = []
    val_accs = []
    for j in range(num_val_batches):
        a = j * batch_size
        z = (j + 1) * batch_size
        x = constant(x_train[a:z])
        y_true = constant(y_train[a:z])
        y_pred = model.forward(x, False)
        loss = categorical_cross_entropy(y_true, y_pred)
        val_losses.append(loss.data.cpu().numpy()[0])
        acc = categorical_accuracy(y_true, y_pred)
        val_accs.append(acc.data.cpu().numpy()[0])
    train_loss = np.array(train_losses).mean()
    train_acc = np.array(train_accs).mean()
    val_loss = np.array(val_losses).mean()
    val_acc = np.array(val_accs).mean()
    print('epoch %4d train %.4f (%.2f%%) val %.4f (%.2f%%)' %
          (i, train_loss, train_acc, val_loss, val_acc))
