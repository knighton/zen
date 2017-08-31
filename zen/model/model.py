import numpy as np

from .. import constant
from ..functional import categorical_accuracy, categorical_cross_entropy
from .. import optim


class Model(object):
    def get_params(self):
        raise NotImplementedError

    def forward(self, x, is_training):
        raise NotImplementedError

    def train_on_batch(self, x, y_true, opt):
        x = constant(x)
        y_true = constant(y_true)
        y_pred = self.forward(x, True)
        loss = categorical_cross_entropy(y_true, y_pred)
        loss_value = loss.data.cpu().numpy()[0]
        loss.backward()
        opt.step()
        acc = categorical_accuracy(y_true, y_pred)
        acc_value = acc.data.cpu().numpy()[0]
        return loss_value, acc_value

    def evaluate_on_batch(self, x, y_true):
        x = constant(x)
        y_true = constant(y_true)
        y_pred = self.forward(x, False)
        loss = categorical_cross_entropy(y_true, y_pred)
        loss_value = loss.data.cpu().numpy()[0]
        acc = categorical_accuracy(y_true, y_pred)
        acc_value = acc.data.cpu().numpy()[0]
        return loss_value, acc_value

    def train_classifier_on_epoch(self, data, opt, batch_size, epoch):
        (x_train, y_train), (x_val, y_val) = data
        num_train_batches = len(x_train) // batch_size
        num_val_batches = len(x_val) // batch_size
        train_losses = []
        train_accs = []
        for j in range(num_train_batches):
            a = j * batch_size
            z = (j + 1) * batch_size
            loss, acc = self.train_on_batch(x_train[a:z], y_train[a:z], opt)
            train_losses.append(loss)
            train_accs.append(acc)
        val_losses = []
        val_accs = []
        for j in range(num_val_batches):
            a = j * batch_size
            z = (j + 1) * batch_size
            loss, acc = self.evaluate_on_batch(x_val[a:z], y_val[a:z])
            val_losses.append(loss)
            val_accs.append(acc)
        train_loss = np.array(train_losses).mean()
        train_acc = np.array(train_accs).mean()
        val_loss = np.array(val_losses).mean()
        val_acc = np.array(val_accs).mean()
        print('epoch %4d train %.4f (%.2f%%) val %.4f (%.2f%%)' %
              (epoch, train_loss, train_acc, val_loss, val_acc))

    def train_classifier(self, data, opt='mvo', batch_size=64, stop=1000):
        opt = optim.get(opt)
        opt.set_params(self.get_params())
        for epoch in range(stop):
            self.train_classifier_on_epoch(data, opt, batch_size, epoch)
