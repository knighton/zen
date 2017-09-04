import numpy as np

from ... import api as Z
from ... import init
from ..layer import Sugar
from .recurrent import RecurrentLayer, RecurrentSpec


class LSTMLayer(RecurrentLayer):
    def __init__(self, input_kernel, recurrent_kernel, bias, out_dim, go, ret):
        super().__init__(out_dim, go, ret)
        self.input_kernel = self.add_param(input_kernel)
        self.recurrent_kernel = self.add_param(recurrent_kernel)
        self.bias = self.add_param(bias)

    def make_initial_internal_state(self, batch_size, out_dim):
        arr = np.zeros((batch_size, self.out_dim)).astype(Z.floatx())
        return Z.constant(arr)

    def step(self, x, prev_state, prev_internal_state):
        a = Z.matmul(x, self.input_kernel) + \
            Z.matmul(prev_state, self.recurrent_kernel) + self.bias
        n = self.out_dim
        i = Z.sigmoid(a[:, :n])
        f = Z.sigmoid(a[:, n:2 * n])
        o = Z.sigmoid(a[:, 2 * n:3 * n])
        g = Z.tanh(a[:, 3 * n:])
        next_internal_state = f * prev_internal_state + i * g
        next_state = o * Z.tanh(next_internal_state)
        return next_state, next_internal_state


class LSTMSpec(RecurrentSpec):
    def __init__(self, out_dim=None, go='forward', ret='all',
                 input_kernel_init='glorot_uniform',
                 recurrent_kernel_init='orthogonal', bias_init='zero'):
        super().__init__(out_dim, go, ret)
        self.input_kernel_init = init.get(input_kernel_init)
        self.recurrent_kernel_init = init.get(recurrent_kernel_init)
        self.bias_init = init.get(bias_init)

    def build(self, in_shape, in_dtype):
        in_dim, out_dim, out_shape = self.get_shapes(in_shape)
        shape = in_dim, 4 * out_dim
        input_kernel = self.input_kernel_init(shape)
        shape = out_dim, 4 * out_dim
        recurrent_kernel = self.recurrent_kernel_init(shape)
        shape = out_dim * 4,
        bias = self.bias_init(shape)
        layer = LSTMLayer(input_kernel, recurrent_kernel, bias, out_dim,
                          self.go, self.ret)
        return layer, out_shape, in_dtype


LSTM = Sugar(LSTMSpec)
BiLSTM = Sugar(LSTMSpec, {'go': 'bidi'})
