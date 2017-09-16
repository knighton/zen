import numpy as np
from time import time

from .. import api as Z
from .. import metric as metric_module
from .. import optim
from .data.dataset import Dataset
from .data.ram_dataset import RamDataset
from .data.training_data import TrainingData
from . import hook as hook_module
from .hook import Hook


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
        if not isinstance(items, (list, tuple)):
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


def _unpack_hook(key, value):
    if value is None or value is False:
        hook = None
    elif value is True:
        hook = getattr(hook_module, key)()
    elif isinstance(value, Hook):
        hook = value
    else:
        hook = getattr(hook_module, key)(value)
    return hook


def _unpack_hooks(defaults, kwargs):
    d = dict(defaults)
    d.update(kwargs)
    ret = []
    for key, value in d.items():
        hook = _unpack_hook(key, value)
        if hook:
            ret.append(hook)
    return ret


class Model(object):
    default_hooks = {
        'stop': 25,
        'verbose': 2,
    }

    def model_params(self):
        raise NotImplementedError

    def model_forward(self, xx, is_training):
        raise NotImplementedError

    def predict_on_batch(self, xx):
        """
        list of np.ndarray -> list of np.ndarray

        Predict on a single batch.
        """
        xx = list(map(Z.constant, xx))
        yy = self.model_forward(xx, False)
        return list(map(Z.to_numpy, yy))

    def predict(self, xx, batch_size=64):
        """
        list of np.ndarray -> list of np.ndarray

        Predict.
        """
        lens = set()
        for x in xx:
            assert isinstance(x, np.ndarray)
            lens.add(len(x))
        assert len(lens) == 1
        assert isinstance(batch_size, int)
        assert 0 < batch_size
        num_samples = list(lens)[0]
        num_batches = num_samples // batch_size
        yy = None
        for i in range(num_batches):
            a = i * batch_size
            z = (i + 1) * batch_size
            ins = []
            for x in xx:
                ins.append(x[a:z])
            outs = self.predict_on_batch(ins)
            if yy is None:
                yy = outs
            else:
                for i, out in enumerate(outs):
                    yy[i] += out[i]
        return yy

    def train_on_batch(self, xx, yy_true, metrics, opt, hooks=None,
                       progress=None):
        """
        Train on a single batch.
        """
        if hooks is None:
            hooks = []
        if progress is None:
            progress = {}

        stop = False
        for hook in hooks:
            if hook.on_train_batch_begin(progress, xx, yy_true):
                stop = True
        if stop:
            results = None
            return results, None

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
        Z.backward(loss_vars, grad_tensors)
        opt.step()

        results = []
        for i, (y_pred, y_true, funcs) in \
                enumerate(zip(yy_pred, yy_true, metrics)):
            loss_value = Z.to_scalar(loss_vars[i])
            values = [loss_value]
            for metric in funcs[1:]:
                value = Z.to_scalar(Z.mean(metric(y_true, y_pred)))
                values.append(value)
            results.append(values)

        stop = False
        for hook in hooks:
            if hook.on_train_batch_end(progress, results):
                stop = True

        return results, stop

    def evaluate_on_batch(self, xx, yy_true, metrics, hooks=None,
                          progress=None):
        """
        Evaluate on a single batch.
        """
        if hooks is None:
            hooks = []
        if progress is None:
            progress = {}

        stop = False
        for hook in hooks:
            if hook.on_eval_batch_begin(progress, xx, yy_true):
                stop = True
        if stop:
            results = None
            return results, None

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

        stop = False
        for hook in hooks:
            if hook.on_eval_batch_end(progress, results):
                stop = True

        return results, stop

    def train_on_epoch(self, data, metrics, opt, batch_size=64, hooks=None,
                       progress=None):
        """
        Train over a single epoch.

        Users should call `train(..., stop=1)` to train for one epoch, not use
        this method directly.  This is called by do_train().
        """
        if hooks is None:
            hooks = []
        if progress is None:
            progress = {}

        stop = False
        for hook in hooks:
            if hook.on_epoch_begin(progress, data):
                stop = True
        if stop:
            results = None
            return results, stop

        num_batches = data.get_num_batches(batch_size)
        train_metrics_per_output = \
            list(map(lambda funcs: [[] for _ in funcs], metrics))
        val_metrics_per_output = \
            list(map(lambda funcs: [[] for _ in funcs], metrics))
        t0 = time()
        for batch, (xx, yy, is_training) in \
                enumerate(data.each_batch(batch_size)):
            sub_progress = dict(progress)
            sub_progress.update({
                'batch': batch,
                'num_batches': num_batches,
            })

            if is_training:
                results, stop = self.train_on_batch(
                    xx, yy, metrics, opt, hooks, sub_progress)
                split_results = train_metrics_per_output
            else:
                results, stop = self.evaluate_on_batch(
                    xx, yy, metrics, hooks, sub_progress)
                split_results = val_metrics_per_output

            if results is not None:
                for i, values in enumerate(results):
                    for j, value in enumerate(values):
                        split_results[i][j].append(value)

            if stop:
                results = None
                return results, stop
        t = time() - t0

        results = {'time': t}
        if progress:
            results['progress'] = progress
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

        stop = False
        for hook in hooks:
            if hook.on_epoch_end(progress, results):
                stop = True

        return results, stop

    def train(self, data, metrics, opt='adam', val=None, batch_size=64,
              start=0, **hooks):
        data = _unpack_training_data(data, val)
        metrics = _unpack_metrics(metrics, data.get_sample_shapes()[1])
        opt = optim.get(opt)
        assert isinstance(start, int)
        assert 0 <= start
        hooks = _unpack_hooks(self.default_hooks, hooks)

        opt.set_params(self.model_params())

        train_kwargs = {
            'data': data,
            'metrics': metrics,
            'opt': opt,
            'epoch_begin': start,
            'hooks': hooks,
        }
        epoch_end_excl = None
        for hook in hooks:
            z = hook.on_train_begin(self, train_kwargs)
            if z is None:
                continue
            if epoch_end_excl is None or z < epoch_end_excl:
                epoch_end_excl = z

        epoch = start
        history = []
        while True:
            progress = {
                'epoch_begin': start,
                'epoch': epoch,
                'epoch_end_excl': epoch_end_excl,
            }
            results, stop = self.train_on_epoch(
                data, metrics, opt, batch_size, hooks, progress)
            if results is not None:
                history.append(results)
            if stop:
                break
            epoch += 1

        for hook in hooks:
            hook.on_train_end(history)

        return history

    def train_regressor(self, data, opt='adam', val=None, batch_size=64,
                        start=0, **hooks):
        """
        Train as a regressor.

        Wrapper around train() that automatically uses mean squared error loss.
        Single output only.
        """
        metrics = 'mean_squared_error',
        return self.train(data, metrics, opt, val, batch_size, start, **hooks)

    def train_classifier(self, data, opt='adam', val=None, batch_size=64,
                         start=0, **hooks):
        """
        Train as a classifier.

        Wrapper around train() that automatically uses cross-entropy loss and
        adds accuracy as a metric.  Single output only.
        """
        metrics = 'cross_entropy', 'accuracy'
        return self.train(data, [metrics], opt, val, batch_size, start, **hooks)
