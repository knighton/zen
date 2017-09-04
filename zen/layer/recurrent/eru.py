from ... import api as Z
from ... import init
from ..layer import Sugar
from .recurrent import RecurrentLayer, RecurrentSpec


class ERULayer(RecurrentLayer):
    def __init__(self, input_kernel, recurrent_kernel, bias, dim, ret):
        super().__init__(dim, ret)
        self.input_kernel = self.add_param(input_kernel)
        self.recurrent_kernel = self.add_param(recurrent_kernel)
        self.bias = self.add_param(bias)

    def step(self, x, prev_state, prev_internal_state):
        next_state = Z.tanh(Z.matmul(x, self.input_kernel) +
                            Z.matmul(prev_state, self.recurrent_kernel) +
                            self.bias)
        return next_state, None


class ERUSpec(RecurrentSpec):
    def __init__(self, dim=None, ret='all', input_kernel_init='glorot_uniform',
                 recurrent_kernel_init='orthogonal', bias_init='zero'):
        super().__init__(dim, ret)
        self.input_kernel_init = init.get(input_kernel_init)
        self.recurrent_kernel_init = init.get(recurrent_kernel_init)
        self.bias_init = init.get(bias_init)

    def build(self, in_shape, in_dtype):
        in_dim, out_dim, out_shape = self.get_shapes(in_shape)
        shape = in_dim, out_dim
        input_kernel = self.input_kernel_init(shape)
        shape = out_dim, out_dim
        recurrent_kernel = self.recurrent_kernel_init(shape)
        shape = out_dim,
        bias = self.bias_init(shape)
        layer = ERULayer(input_kernel, recurrent_kernel, bias, out_dim,
                         self.ret)
        return layer, out_shape, in_dtype


ERU = Sugar(ERUSpec)
