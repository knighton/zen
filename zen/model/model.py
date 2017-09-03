import numpy as np
from time import time

from .. import api as Z
from .. import metric as metric_module
from .. import optim


def _bin_or_cat(y_sample_shape, if_bin, if_cat):
    return if_bin if y_sample_shape in {(), (1,)} else if_cat


def _unpack_metrics(metrics, y_sample_shape):
    ret = []
    for metric in metrics:
        if metric in {'xe', 'cross_entropy'}:
            metric = _bin_or_cat(y_sample_shape, 'binary_cross_entropy',
                                 'categorical_cross_entropy')
        elif metric in {'acc', 'accuracy'}:
            metric = _bin_or_cat(y_sample_shape, 'binary_accuracy',
                                 'categorical_accuracy')
        else:
            pass
        metric = metric_module.get(metric)
        ret.append(metric)
    return ret


class Model(object):
    def get_params(self):
        raise NotImplementedError

    def forward(self, x, is_training):
        raise NotImplementedError

    def train_on_batch(self, x, y_true, metrics, opt):
        x = Z.constant(x)
        y_true = Z.constant(y_true)
        with Z.autograd_record():
            y_pred = self.forward(x, True)
            loss_var = Z.mean(metrics[0](y_true, y_pred))
        loss_var.backward()
        opt.step()
        metric_values = [Z.to_scalar(loss_var)]
        for i in range(1, len(metrics)):
            metric = metrics[i]
            metric_var = Z.mean(metric(y_true, y_pred))
            metric_values.append(Z.to_scalar(metric_var))
        return metric_values

    def evaluate_on_batch(self, x, y_true, metrics):
        x = Z.constant(x)
        y_true = Z.constant(y_true)
        y_pred = self.forward(x, False)
        metric_values = []
        for metric in metrics:
            metric_var = Z.mean(metric(y_true, y_pred))
            metric_values.append(Z.to_scalar(metric_var))
        return metric_values

    def train_on_epoch(self, data, metrics, opt, batch_size, epoch):
        t0 = time()
        (x_train, y_train), (x_val, y_val) = data
        num_train_batches = len(x_train) // batch_size
        num_val_batches = len(x_val) // batch_size
        train_metric_value_lists = [[] for _ in metrics]
        for i in range(num_train_batches):
            a = i * batch_size
            z = (i + 1) * batch_size
            metric_values = self.train_on_batch(
                x_train[a:z], y_train[a:z], metrics, opt)
            for i, metric_value in enumerate(metric_values):
                train_metric_value_lists[i].append(metric_value)
        val_metric_value_lists = [[] for _ in metrics]
        for i in range(num_val_batches):
            a = i * batch_size
            z = (i + 1) * batch_size
            metric_values = self.evaluate_on_batch(
                x_train[a:z], y_train[a:z], metrics)
            for i, metric_value in enumerate(metric_values):
                val_metric_value_lists[i].append(metric_value)
        train_metric_values = []
        for values in train_metric_value_lists:
            train_metric_values.append(np.array(values).mean())
        val_metric_values = []
        for values in val_metric_value_lists:
            val_metric_values.append(np.array(values).mean())
        t = time() - t0
        print('epoch %4d took %.3f sec train %s val %s' %
              (epoch, t, train_metric_values, val_metric_values))

    def train(self, data, metrics, opt='mvo', batch_size=64, stop=1000):
        y_sample_shape = data[0][1].shape[1:]
        metrics = _unpack_metrics(metrics, y_sample_shape)
        opt = optim.get(opt)
        opt.set_params(self.get_params())
        for epoch in range(stop):
            self.train_on_epoch(data, metrics, opt, batch_size, epoch)

    def train_regressor(self, data, opt='mvo', batch_size=64, stop=1000):
        """
        Train as a regressor.

        Wrapper around train() that automatically uses mean squared error loss.
        Single output only.
        """
        metrics = 'mean_squared_error',
        return self.train(data, metrics, opt, batch_size, stop)

    def train_classifier(self, data, opt='mvo', batch_size=64, stop=1000):
        """
        Train as a classifier.

        Wrapper around train() that automatically uses cross-entropy loss and
        adds accuracy as a metric.  Single output only.
        """
        metrics = 'cross_entropy', 'accuracy'
        return self.train(data, metrics, opt, batch_size, stop)
