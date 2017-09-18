from ... import func as Z
from ..base import Node


class Input(Node):
    """
    Neural network input node.
    """

    def __init__(self, shape, dtype=None):
        """
        Create an input node given shape and optionally dtype.

        >>> image = Input((3, 32, 32))
        >>> tokens = Input((64,), dtype='int64')
        """
        super().__init__()
        Z.check_shape(shape, None, 'shape')
        dtype = dtype or Z.floatx()
        self._out_shape = shape
        self._out_dtype = dtype
        self._is_built = False

    def try_to_build(self):
        if self._is_built:
            return True
        for node in self._out_nodes:
            node.try_to_build()
        self._is_built = True
        return True

    def is_built(self):
        return self._is_built

    def params(self):
        return []

    def feed(self, data, is_training):
        """
        Receive and propagate input through the network.
        """
        assert Z.get_shape(data)[1:] == self._out_shape
        assert Z.get_dtype(data) == self._out_dtype
        self._out_data = data
        for node in self._out_nodes:
            node.in_node_is_ready(is_training)
