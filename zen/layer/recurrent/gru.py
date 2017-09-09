from ... import api as Z
from ... import init
from ..base import Sugar
from .recurrent import RecurrentLayer, RecurrentSpec


class GRULayer(RecurrentLayer):
    def __init__(self, input_kernel, recurrent_kernel, bias, out_dim, go, ret):
        super().__init__(out_dim, go, ret)
        self.input_kernel = self.add_param(input_kernel)
        self.recurrent_kernel = self.add_param(recurrent_kernel)
        self.bias = self.add_param(bias)

        n = out_dim
        self.rz_input_kernel = self.input_kernel[:, :2 * n]
        self.rz_recurrent_kernel = self.recurrent_kernel[:, :2 * n]
        self.rz_bias = self.bias[:2 * n]

        self.h_input_kernel = self.input_kernel[:, 2 * n:]
        self.h_recurrent_kernel = self.recurrent_kernel[:, 2 * n:]
        self.h_bias = self.bias[2 * n:]

    def step(self, x, prev_state, prev_internal_state):
        rz = Z.sigmoid(Z.matmul(x, self.rz_input_kernel) +
                       Z.matmul(prev_state, self.rz_recurrent_kernel) +
                       self.rz_bias)
        n = self.out_dim
        r = rz[:, :n]
        z = rz[:, n:2 * n]
        h = Z.tanh(Z.matmul(x, self.h_input_kernel) +
                   Z.matmul(r * prev_state, self.h_recurrent_kernel) +
                   self.h_bias)
        next_state = z * prev_state + (1. - z) * h
        return next_state, None


class GRUSpec(RecurrentSpec):
    def __init__(self, out_dim=None, go='forward', ret='all',
                 input_kernel_init='glorot_uniform',
                 recurrent_kernel_init='orthogonal', bias_init='zero'):
        super().__init__(out_dim, go, ret)
        self.input_kernel_init = init.get(input_kernel_init)
        self.recurrent_kernel_init = init.get(recurrent_kernel_init)
        self.bias_init = init.get(bias_init)

    def build(self, in_shape, in_dtype):
        in_dim, out_dim, out_shape = self.get_shapes(in_shape)
        shape = in_dim, 3 * out_dim
        input_kernel = self.input_kernel_init(shape)
        shape = out_dim, 3 * out_dim
        recurrent_kernel = self.recurrent_kernel_init(shape)
        shape = out_dim * 3,
        bias = self.bias_init(shape)
        layer = GRULayer(input_kernel, recurrent_kernel, bias, out_dim, self.go,
                         self.ret)
        return layer, out_shape, in_dtype


GRU = Sugar(GRUSpec)
BiGRU = Sugar(GRUSpec, {'go': 'bidi'})
