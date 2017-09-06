import numpy as np


class Dataset(object):
    """
    A collection of (X, Y) samples.  Iterate over them using each_batch().

    For example:
        [images] -> [digits] for MNIST
        [images, questions] -> [answers] for visual question answering
    """

    def get_num_samples(self):
        """
        -> int
        """
        raise NotImplementedError

    def get_sample(self, index):
        """
        sample index -> (tuple of np.ndarray, tuple of np.ndarray)
        """
        raise NotImplementedError

    def get_sample_shapes(self):
        """
        -> (tuple of shape, tuple of shape)
        """
        xx, yy = self.get_sample(0)
        xx = tuple(map(lambda x: x.shape, xx))
        yy = tuple(map(lambda y: y.shape, yy))
        return xx, yy

    def get_sample_dtypes(self):
        """
        -> (tuple of dtype, tuple of dtype)
        """
        xx, yy = self.get_sample(0)
        xx = tuple(map(lambda x: x.dtype.name, xx))
        yy = tuple(map(lambda y: y.dtype.name, yy))
        return xx, yy

    def get_num_batches(self, samples_per_batch):
        """
        -> int
        """
        return self.get_num_samples() // samples_per_batch

    def each_batch(self, samples_per_batch):
        """
        yields (tuple of np.ndarray, tuple of np.ndarray)
        """
        num_samples = self.get_num_samples()
        num_batches = num_samples // samples_per_batch
        sample_indexes = np.arange(num_samples)
        np.random.shuffle(sample_indexes)
        for batch_index in range(num_batches):
            a = batch_index * samples_per_batch
            z = (batch_index + 1) * samples_per_batch
            xxx = None
            yyy = None
            for sample_index in sample_indexes[a:z]:
                sample = self.get_sample(sample_index)
                if not xxx:
                    xxx = [[] for _ in sample[0]]
                for i, x in enumerate(sample[0]):
                    xxx[i].append(x)
                if not yyy:
                    yyy = [[] for _ in sample[1]]
                for i, y in enumerate(sample[1]):
                    yyy[i].append(y)
            for i, xx in enumerate(xxx):
                xxx[i] = np.array(xx).astype(xx[0].dtype)
            for i, yy in enumerate(yyy):
                yyy[i] = np.array(yy).astype(yy[0].dtype)
            yield xxx, yyy
