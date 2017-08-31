import numpy as np

from .. import constant
from ..functional import categorical_accuracy, categorical_cross_entropy


class Model(object):
    def get_params(self):
        raise NotImplementedError

    def forward(self, x, is_training):
        raise NotImplementedError

    def train_classifier_on_epoch(self, data, opt, batch_size, epoch):
        (x_train, y_train), (x_val, y_val) = data
        num_train_batches = len(x_train) // batch_size
        num_val_batches = len(x_val) // batch_size
        train_losses = []
        train_accs = []
        for j in range(num_train_batches):
            a = j * batch_size
            z = (j + 1) * batch_size
            x = constant(x_train[a:z])
            y_true = constant(y_train[a:z])
            y_pred = self.forward(x, True)
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
            y_pred = self.forward(x, False)
            loss = categorical_cross_entropy(y_true, y_pred)
            val_losses.append(loss.data.cpu().numpy()[0])
            acc = categorical_accuracy(y_true, y_pred)
            val_accs.append(acc.data.cpu().numpy()[0])
        train_loss = np.array(train_losses).mean()
        train_acc = np.array(train_accs).mean()
        val_loss = np.array(val_losses).mean()
        val_acc = np.array(val_accs).mean()
        print('epoch %4d train %.4f (%.2f%%) val %.4f (%.2f%%)' %
              (epoch, train_loss, train_acc, val_loss, val_acc))

    def train_classifier(self, data, opt, batch_size=64, stop=1000):
        opt.set_params(self.get_params())
        for epoch in range(stop):
            self.train_classifier_on_epoch(data, opt, batch_size, epoch)
