import numpy as np
from time import time
import torch

from .. import api as Z
from .. import metric as metric_module
from .. import optim
from .data.dataset import Dataset
from .data.ram_dataset import RamDataset
from .data.training_data import TrainingData


def _unpack_training_data(data, val=None):
    """
    Unpack the given training data.

    It can take different forms:
    * TrainingData: we already have a training data object.
    * Given numpy arrays and `val` fraction: perform our own train/val split.
      * MNIST: np.ndarray, np.ndarray
      * Visual question answering: (np.ndarray, np.ndarray), np.ndarray
    * No `val`: the data is a 2-tuple of (train split, val split).
      * MNIST: (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)
      * Also MNIST: RamDataset, RamDataset
    """
    if isinstance(data, TrainingData):
        assert val is None
        return data

    if val is not None:
        x, y = data
        return TrainingData.from_x_y(x, y, val)

    train, val = data
    if not isinstance(train, Dataset):
        xx, yy = train
        train = RamDataset(xx, yy)
    if not isinstance(val, Dataset):
        xx, yy = val
        val = RamDataset(xx, yy)
    return TrainingData(train, val)


def _bin_or_cat(y_sample_shape, if_bin, if_cat):
    return if_bin if y_sample_shape in {(), (1,)} else if_cat


def _unpack_metric(metric, y_sample_shape):
    if metric in {'xe', 'cross_entropy'}:
        metric = _bin_or_cat(y_sample_shape, 'binary_cross_entropy',
                             'categorical_cross_entropy')
    elif metric in {'acc', 'accuracy'}:
        metric = _bin_or_cat(y_sample_shape, 'binary_accuracy',
                             'categorical_accuracy')
    else:
        pass
    return metric_module.get(metric)


def _unpack_metrics(metrics, out_shapes):
    rrr = []
    for i, items in enumerate(metrics):
        if not hasattr(items, '__getitem__'):
            items = [items]
        out_shape = out_shapes[i]
        loss = _unpack_metric(items[0], out_shape)
        metrics = []
        for item in items[1:]:
            metric = _unpack_metric(item, out_shape)
            metrics.append(metric)
        rr = [loss] + metrics
        rrr.append(rr)
    return rrr


class Model(object):
    def model_params(self):
        raise NotImplementedError

    def model_forward(self, xx, is_training):
        raise NotImplementedError

    def train_on_batch(self, xx, yy_true, metrics, opt):
        for i, x in enumerate(xx):
            xx[i] = Z.constant(x)
        for i, y_true in enumerate(yy_true):
            yy_true[i] = Z.constant(y_true)

        is_training = True
        loss_vars = []
        with Z.autograd_record():
            yy_pred = self.model_forward(xx, is_training)
            for y_pred, y_true, funcs in zip(yy_pred, yy_true, metrics):
                loss = funcs[0]
                loss_var = Z.mean(loss(y_true, y_pred))
                loss_vars.append(loss_var)

        grad_tensors = []
        for y_pred in yy_pred:
            grad_tensor = Z.tensor(np.ones(1).astype(Z.floatx()))
            grad_tensors.append(grad_tensor)
        torch.autograd.backward(loss_vars, grad_tensors)
        opt.step()

        results = []
        for i, (y_pred, y_true, funcs) in \
                enumerate(zip(yy_pred, yy_true, metrics)):
            loss_value = Z.to_scalar(loss_vars[i])
            values = [loss_value]
            for metric in funcs[1:]:
                value = Z.to_scalar(metric(y_true, y_pred))
                values.append(value)
            results.append(values)
        return results

    def evaluate_on_batch(self, xx, yy_true, metrics):
        for i, x in enumerate(xx):
            xx[i] = Z.constant(x)
        for i, y_true in enumerate(yy_true):
            yy_true[i] = Z.constant(y_true)

        is_training = False
        yy_pred = self.model_forward(xx, is_training)
        results = []
        for i, (y_pred, y_true, metrics) in \
                enumerate(zip(yy_pred, yy_true, metrics)):
            values = []
            for j, metric in enumerate(metrics):
                var = Z.mean(metric(y_true, y_pred))
                values.append(Z.to_scalar(var))
            results.append(values)
        return results

    def train_on_epoch(self, data, metrics, opt, batch_size, epoch):
        t0 = time()
        num_batches = data.get_num_batches(batch_size)
        train_metrics_per_output = \
            list(map(lambda funcs: [[] for _ in funcs], metrics))
        val_metrics_per_output = \
            list(map(lambda funcs: [[] for _ in funcs], metrics))
        for batch, (xx, yy, is_training) in \
                enumerate(data.each_batch(batch_size)):
            if is_training:
                results = self.train_on_batch(xx, yy, metrics, opt)
                split = train_metrics_per_output
            else:
                results = self.evaluate_on_batch(xx, yy, metrics)
                split = val_metrics_per_output
            for i, values in enumerate(results):
                for j, value in enumerate(values):
                    split[i][j].append(value)
        t = time() - t0
        results = {}
        results['epoch'] = epoch
        results['time'] = t
        mean = lambda ff: sum(ff) / len(ff)
        results['train'] = []
        for i, metric_value_lists in enumerate(train_metrics_per_output):
            means = []
            for values in metric_value_lists:
                means.append(mean(values))
            results['train'].append(means)
        if val_metrics_per_output[0][0]:
            results['val'] = []
            for i, metric_value_lists in enumerate(val_metrics_per_output):
                means = []
                for values in metric_value_lists:
                    means.append(mean(values))
                results['val'].append(means)
        _ss = lambda ff: ' '.join(map(lambda f: '%.3f' % f, ff))
        _sss = lambda fff: '|'.join(map(_ss, fff))
        print('epoch %4d  %.3f sec  train %s  val %s' %
              (results['epoch'], results['time'], _sss(results['train']),
               _sss(results['val'])))

    def train(self, data, metrics, opt='adam', val=None, batch_size=64,
              stop=1000):
        data = _unpack_training_data(data, val)
        metrics = _unpack_metrics(metrics, data.get_sample_shapes()[1])
        opt = optim.get(opt)
        opt.set_params(self.model_params())
        for epoch in range(stop):
            self.train_on_epoch(data, metrics, opt, batch_size, epoch)

    def train_regressor(self, data, opt='adam', val=None, batch_size=64,
                        stop=1000):
        """
        Train as a regressor.

        Wrapper around train() that automatically uses mean squared error loss.
        Single output only.
        """
        metrics = 'mean_squared_error',
        return self.train(data, metrics, opt, val, batch_size, stop)

    def train_classifier(self, data, opt='adam', val=None, batch_size=64,
                         stop=1000):
        """
        Train as a classifier.

        Wrapper around train() that automatically uses cross-entropy loss and
        adds accuracy as a metric.  Single output only.
        """
        metrics = 'cross_entropy', 'accuracy'
        return self.train(data, [metrics], opt, val, batch_size, stop)
