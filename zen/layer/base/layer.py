from ... import api as Z


class Layer(object):
    """
    Thing that applies differentiable changes to variables.

    They live inside Nodes, which connect their execution together into neural
    networks.
    """

    def __init__(self):
        super().__init__()
        self._params = []

    def add_param(self, arr, trainable=True):
        """
        np.ndarray, trainable ->

        Make a variable or constant parameter, adding variables to the list of
        things returned to the optimizer to train.
        """
        if trainable:
            param = Z.variable(arr)
            self._params.append(param)
        else:
            param = Z.constant(arr)
        return param

    def params(self):
        """
        -> variables

        Collect all the variables for the optimizer to train.
        """
        return self._params

    def forward_multi_input(self, xx, is_training):
        """
        variables, is_training -> variables

        Perform the work.
        """
        raise NotImplementedError


class Transform(Layer):
    """
    A single input, single output layer (the normal case).
    """

    def forward(self, x, is_training):
        """
        variable, is_training -> variable

        Perform the work.
        """
        raise NotImplementedError

    def forward_multi_input(self, xx, is_training):
        """
        variables, is_training -> variables

        Perform the work.  Wraps forward().
        """
        assert len(xx) == 1
        x = xx[0]
        return self.forward(x, is_training)


class Merge(Layer):
    """
    A multiple input, single input layer.
    """

    def forward_multi_input(self, xx, is_training):
        """
        variables, is_training -> variables

        Perform the work.  Merge layers have no forward().
        """
        assert len(xx) == 1
        raise NotImplementedError
