import numpy as np

from ... import api as Z
from ... import init
from ..layer import Layer, Spec


class RecurrentLayer(Layer):
    def __init__(self, dim, ret):
        super().__init__()
        Z.check_dim(dim)
        self.out_dim = dim
        assert ret in {'all', 'last'}
        self.ret = ret

    def make_initial_internal_state(self, num_samples, out_dim):
        """
        -> internal_state
        """
        return None

    def step(self, x, prev_state, prev_internal_state):
        """
        x, prev_state, prev_internal_state -> next_state, next_internal_state
        """
        raise NotImplementedError

    def forward(self, x, is_training):
        num_samples, num_out_dim, num_timesteps = Z.get_shape(x)
        arr = np.zeros((num_samples, self.out_dim)).astype(Z.floatx())
        initial_state = Z.constant(arr)
        states = [initial_state]
        initial_internal_state = self.make_initial_internal_state(
            num_samples, self.out_dim)
        internal_states = [initial_internal_state]
        for timestep in range(num_timesteps):
            x_step = x[:, :, timestep]
            next_state, next_internal_state = \
                self.step(x_step, states[-1], internal_states[-1])
            states.append(next_state)
            internal_states.append(next_internal_state)
        for i in range(1, len(states)):
            states[i] = Z.expand_dims(states[i], 2)
        states = Z.concat(states[1:], 2)
        if self.ret == 'all':
            return states
        elif self.ret == 'last':
            return Z.squeeze(states[:, :, -1])
        else:
            assert False


class RecurrentSpec(Spec):
    def __init__(self, dim=None, ret='all'):
        super().__init__()
        if dim is not None:
            Z.check_dim(dim)
        assert ret in {'all', 'last'}
        self.out_dim = dim
        self.ret = ret

    def get_shapes(self, in_shape):
        Z.verify_input_ndim(1, in_shape)
        in_dim = in_shape[0]
        if self.out_dim is None:
            out_dim = in_dim
        else:
            out_dim = self.out_dim
        if self.ret == 'all':
            in_dim, num_timesteps = in_shape
            out_shape = out_dim, num_timesteps
        elif self.ret == 'last':
            out_shape = out_dim,
        else:
            assert False
        return in_dim, out_dim, out_shape
