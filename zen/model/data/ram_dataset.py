import numpy as np

from .dataset import Dataset


class RamDataset(Dataset):
    def __init__(self, xx, yy):
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
        self.xx = xx
        self.yy = yy

    def get_num_samples(self):
        return len(self.xx[0])

    def get_sample(self, index):
        xx = []
        for x in self.xx:
            xx.append(x[index])
        yy = []
        for y in self.yy:
            yy.append(y[index])
        return xx, yy
