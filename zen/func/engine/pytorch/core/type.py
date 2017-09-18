from torch.autograd import Variable
from torch import _TensorBase

from .. import base as B


class DataTypeConverter(object):
    def __init__(self):
        data = [
            ('float16', None, 'torch.cuda.HalfTensor'),
            ('float32', 'torch.FloatTensor', 'torch.cuda.FloatTensor'),
            ('float64', 'torch.DoubleTensor', 'torch.cuda.DoubleTensor'),
            ('uint8', 'torch.ByteTensor', 'torch.cuda.ByteTensor'),
            ('int8', 'torch.CharTensor', 'torch.cuda.CharTensor'),
            ('int16', 'torch.ShortTensor', 'torch.cuda.ShortTensor'),
            ('int32', 'torch.IntTensor', 'torch.cuda.IntTensor'),
            ('int64', 'torch.LongTensor', 'torch.cuda.LongTensor'),
        ]

        self.numpy_dtypes = set()
        self.numpy2cpu_tensor = {}
        self.numpy2gpu_tensor = {}
        self.tensor2numpy = {}
        for dtype, cpu, gpu in data:
            assert dtype
            self.numpy_dtypes.add(dtype)
            if cpu:
                self.tensor2numpy[cpu] = dtype
                self.numpy2cpu_tensor[dtype] = cpu
            if gpu:
                self.tensor2numpy[gpu] = dtype
                self.numpy2gpu_tensor[dtype] = gpu


_DTC = DataTypeConverter()


def cast(x, dtype=None):
    dtype = dtype or B.floatx()
    if x.is_cuda:
        t = _DTC.numpy2gpu_tensor[dtype]
    else:
        t = _DTC.numpy2cpu_tensor[dtype]
    return x.type(t)


def dtype(x):
    if isinstance(x, Variable):
        tensor = x.data
    elif isinstance(x, _TensorBase):
        tensor = x
    else:
        assert False
    return _DTC.tensor2numpy[tensor.type()]
