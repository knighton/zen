from copy import deepcopy

from ... import api as Z


class Node(object):
    """
    A node of a neural network.

    They consist of input and non-input nodes (Inputs and LayerNodes).
    """

    def __init__(self):
        self._out_shape = None
        self._out_dtype = None
        self._out_data = None
        self._out_nodes = []

    def out_shape(self):
        """
        -> shape (must be built)
        """
        return self._out_shape

    def out_dtype(self):
        """
        -> dtype (must be built)
        """
        return self._out_dtype

    def out_data(self):
        """
        -> data (must be forward()'ed)
        """
        return self._out_data

    def add_out_node(self, node):
        """
        node ->
        """
        self._out_nodes.append(node)

    def out_nodes(self):
        """
        -> node
        """
        return self._out_nodes

    def try_to_build(self):
        raise NotImplementedError

    def is_built(self):
        raise NotImplementedError

    def params(self):
        raise NotImplementedError


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


class LayerNode(Node):
    """
    Neural network non-input node (this is the normal case).
    """

    def __init__(self, spec, in_nodes=None):
        super().__init__()
        if in_nodes:
            for node in in_nodes:
                node.add_out_node(self)
        self._in_nodes = in_nodes
        self._num_ready_in_nodes = 0
        self._spec = spec
        self._layer = None

    def __call__(self, *in_nodes):
        """
        Return a copy of ourself that is connected to the given feed nodes.
        This is how graphs are constructed.
        """
        assert not self._in_nodes
        return LayerNode(deepcopy(self._spec), in_nodes)

    def try_to_build(self):
        """
        Try to construct the internal layer of a node given the shapes and
        dtypes of its input nodes.  Tries to build its output nodes.

        Returns true if this node could be built (output nodes will fail if not
        all inputs are built yet during graph building).
        """
        assert self._in_nodes, 'Tried to build an internal node with no inputs.'
        in_shapes = []
        in_dtypes = []
        for node in self._in_nodes:
            shape = node.out_shape()
            if shape is None:
                return False
            in_shapes.append(shape)
            dtype = node.out_dtype()
            if dtype is None:
                return False
            in_dtypes.append(dtype)
        self._layer, self._out_shape, self._out_dtype = \
            self._spec.build_multi_input(in_shapes, in_dtypes)
        for node in self._out_nodes:
            node.try_to_build()
        return True

    def is_built(self):
        return self._layer is not None

    def params(self):
        """
        Build the node if not built, then collect the node's trainable
        parameters for the optimizer.
        """
        assert self._layer, \
            'Not all input nodes have been built (the graph is missing an ' + \
            'input or inputs).'
        return self._layer.params()

    def in_node_is_ready(self, is_training):
        """
        Receive notification that one of our input nodes has data.  If they all
        do, perform a forward pass and notify the nodes that we feed into.
        """
        assert self._in_nodes, \
            'Called in_node_is_ready() on a node with no inputs.'

        assert self._layer, \
            'Not all input nodes have been built (the graph is missing an ' + \
            'input or inputs).'

        self._num_ready_in_nodes += 1
        if self._num_ready_in_nodes < len(self._in_nodes):
            return

        xx = []
        for node in self._in_nodes:
            x = node.out_data()
            assert x is not None
            xx.append(x)
        self._out_data = self._layer.forward_multi_input(xx, is_training)
        for node in self._out_nodes:
            node.in_node_is_ready(is_training)
        self._num_ready_in_nodes = 0
