from copy import deepcopy

from ..arch.vee import Vee


class Node(Vee):
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


class InteriorNode(Node):
    """
    A non-input node (the normal case).
    """

    def __init__(self):
        super().__init__()
        self._in_nodes = None
        self._num_ready_in_nodes = 0

    def _gather_shapes_dtypes_for_build(self):
        assert self._in_nodes, 'Tried to build an internal node with no inputs.'
        in_shapes = []
        in_dtypes = []
        for node in self._in_nodes:
            shape = node.out_shape()
            if shape is None:
                return False, None, None
            in_shapes.append(shape)
            dtype = node.out_dtype()
            if dtype is None:
                return False, None, None
            in_dtypes.append(dtype)
        return True, in_shapes, in_dtypes

    def in_nodes(self):
        return self._in_nodes

    def to_spec_or_specs(self):
        raise NotImplementedError


class LayerNode(InteriorNode):
    """
    Neural network node wrapping a single layer.
    """

    def __init__(self, spec, in_nodes=None):
        super().__init__()
        if in_nodes:
            for node in in_nodes:
                node.add_out_node(self)
        self._in_nodes = in_nodes
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
        can_build, in_shapes, in_dtypes = self._gather_shapes_dtypes_for_build()
        if not can_build:
            return False
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

    def to_spec_or_specs(self):
        return self._spec
