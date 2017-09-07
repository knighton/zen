import numpy as np

from .dataset import Dataset
from .ram_dataset import RamDataset


class TrainingData(object):
    def __init__(self, train, val=None):
        assert isinstance(train, Dataset)
        if val is not None:
            assert isinstance(val, Dataset)
            assert train.get_sample_shapes() == val.get_sample_shapes()
            assert train.get_sample_dtypes() == val.get_sample_dtypes()
        self.train = train
        self.val = val

    def get_num_batches(self, samples_per_batch):
        """
        -> int
        """
        if self.val:
            return self.train.get_num_batches(samples_per_batch) + \
                self.val.get_num_batches(samples_per_batch)
        else:
            return self.train.get_num_batches(samples_per_batch)

    def get_sample_shapes(self):
        """
        -> (tuple of shape, tuple of shape)
        """
        return self.train.get_sample_shapes()

    def get_sample_dtypes(self):
        """
        -> (tuple of dtype, tuple of dtype)
        """
        return self.train.get_sample_dtypes()

    def each_batch(self, samples_per_batch):
        """
        yields (tuple of np.ndarray, tuple of np.ndarray, bool)
        """
        is_trains = [1] * self.train.get_num_batches(samples_per_batch)
        if self.val:
            is_trains += [0] * self.val.get_num_batches(samples_per_batch)
        np.random.shuffle(is_trains)
        each_train_batch = self.train.each_batch(samples_per_batch)
        each_val_batch = self.val.each_batch(samples_per_batch)
        for is_train in is_trains:
            if is_train:
                yield next(each_train_batch) + (True,)
            else:
                yield next(each_val_batch) + (False,)

    @classmethod
    def from_x_y(cls, xx, yy, val):
        if isinstance(xx, np.ndarray):
            xx = [xx]
        else:
            for x in xx:
                assert isinstance(x, np.ndarray)
        if isinstance(yy, np.ndarray):
            yy = [yy]
        else:
            for y in yy:
                assert isinstance(y, np.ndarray)
        assert len(xx[0])
        assert len(set(list(map(len, xx)) + list(map(len, yy)))) == 1
        assert 0. <= val < 1.
        train_xx = [[] for _ in xx]
        train_yy = [[] for _ in yy]
        val_xx = [[] for _ in xx]
        val_yy = [[] for _ in yy]
        for i in range(len(xx[0])):
            if val < np.random.uniform():
                for j, x in enumerate(xx):
                    train_xx[j].append(x[i])
                for j, y in enumerate(yy):
                    train_yy[j].append(y[i])
            else:
                for j, x in enumerate(xx):
                    val_xx[j].append(x[i])
                for j, y in enumerate(yy):
                    val_yy[j].append(y[i])
        for i, x in enumerate(train_xx):
            train_xx[i] = np.stack(x)
        for i, y in enumerate(train_yy):
            train_yy[i] = np.stack(y)
        train = RamDataset(train_xx, train_yy)
        if val_xx:
            for i, x in enumerate(val_xx):
                val_xx[i] = np.stack(x)
            for i, y in enumerate(val_yy):
                val_yy[i] = np.stack(y)
            val = RamDataset(val_xx, val_yy)
        else:
            val = None
        return TrainingData(train, val)
