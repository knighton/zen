from copy import deepcopy

from ...model.model import Model
from ..base import InteriorNode, Layer, Node, Spec, Sugar
from .input import Input


def _desugar(node):
    if isinstance(node, Sugar):
        return node()
    elif isinstance(node, Node):
        return node
    else:
        assert False, \
            ('Tried to create a Sequence with an invalid object (got a(n) ' +
             '%s: %s) (need a Node or Sugar).') % \
            (node.__class__.__name__, node)


def _is_input_or_has_in_nodes(node):
    if isinstance(node, Input):
        return True
    elif isinstance(node, InteriorNode):
        return node.in_nodes() is not None
    else:
        assert False, \
            'Not a Node (got a(n) %s: %s).' % (node.__class__.__name__, node)


def _find_model_input_nodes_sub(arg):
    if isinstance(arg, Input):
        yield arg
    elif isinstance(arg, InteriorNode):
        if arg._in_nodes is not None:
            for node in arg._in_nodes:
                for input_ in _find_model_input_nodes_sub(node):
                    yield input_
    else:
        assert False, 'Not a Node.'


def _find_model_input_nodes(in_nodes):
    inputs = []
    ids_seen = set()
    for node in in_nodes:
        for input_ in _find_model_input_nodes_sub(node):
            id_ = id(input_)
            if id_ in ids_seen:
                continue
            inputs.append(input_)
            ids_seen.add(id_)
    return inputs


def _show_recursive(arg):
    if isinstance(arg, Spec):
        return arg.__class__.__name__
    elif isinstance(arg, (list, tuple)):
        ss = []
        for spec in arg:
            ss.append(_show_recursive(spec))
        return ss
    else:
        assert False, \
            'Not a Spec or list of Specs (got a(n) %s: %s).' % \
            (arg.__class__.__name__, arg)


def _show(arg):
    x = _show_recursive(arg)
    import json
    print(json.dumps(x, indent=4, sort_keys=True))


def _build_recursive(arg, in_shapes, in_dtypes):
    if isinstance(arg, Spec):
        layer, out_shape, out_dtype = \
            arg.build_multi_input(in_shapes, in_dtypes)
        return layer, out_shape, out_dtype
    elif isinstance(arg, (list, tuple)):
        layers = []
        for spec in arg:
            layer, out_shape, out_dtype = \
                _build_recursive(spec, in_shapes, in_dtypes)
            layers.append(layer)
            in_shapes = [out_shape]
            in_dtypes = [out_dtype]
        return layers, out_shape, out_dtype
    else:
        assert False, \
            'Not a Spec or list of Specs (got a(n) %s: %s).' % \
            (arg.__class__.__name__, arg)


def _params_recursive(arg):
    if isinstance(arg, Layer):
        params = arg.params()
    elif isinstance(arg, (list, tuple)):
        params = []
        for layer in arg:
            params += _params_recursive(layer)
    else:
        assert False, \
            'Not a Layer or list of Layers (got a(n) %s: %s).' % \
            (arg.__class__.__name__, arg)
    return params


def _forward_recursive(arg, xx, is_training):
    if isinstance(arg, Layer):
        x = arg.forward_multi_input(xx, is_training)
    elif isinstance(arg, (list, tuple)):
        for layer in arg:
            x = _forward_recursive(layer, xx, is_training)
            xx = [x]
    else:
        assert False, \
            'Not a Layer or list of Layers (got a(n) %s: %s).' % \
            (arg.__class__.__name__, arg)
    return x


class Sequence(InteriorNode, Model):
    """
    Neural network node wrapping a sequence of layers.
    """

    def __init__(self, *nodes, in_nodes=None):
        """
        Create a sequence node.  Several different valid possibilities:

        1. Orphan (disconnected) sequences (must be connected before use):

            >>> x = Sequence(Conv(64), Conv(64), Conv(64))

            >>> x = Conv(64) > Conv(64) > Conv(64) > Z

        2. Connected sequences (sequence syntax).  Connecting to previous layers
           this way is easier.

            >>> x = Sequence(Input((3, 32, 32)), Conv(64), Conv(64), Conv(64))

            >>> x = Input((3, 32, 32) > Conv(64) > Conv(64) > Conv(64) > Z

            Note: can be recursive.

            >>> image = Input((3, 32, 32))
            >>> x = Sequence(image, Conv(64), Conv(64), Conv(64))
            >>> x = Sequence(x, Conv(64), Conv(64), Conv(64))

            >>> image = Input((3, 32, 32))
            >>> x = image > Conv(64) > Conv(64) > Conv(64) > Z
            >>> x = x > Conv(64) > Conv(64) > Conv(64) > Z

            Also note: can reuse the first node (as a result, the first node
            must live outside the sequence, while all subsequent nodes have
            their specs/layers extracted).

            >>> image = Input((3, 32, 32))
            >>> x = image > Conv(64) > Conv(64) > Conv(64) > Z
            >>> _3x3 = x > Conv(64) > Conv(64) > Conv(64) > Z
            >>> _5x5 = x > Conv(64, 5, 2) > Conv(64, 5, 2) > Conv(64, 5, 2) > Z
            >>> x = Concat()(_3x3, _5x5) > Flatten > Dense(1) > Sigmoid > Z

        3. Connected sequences (graph syntax).  Connecting this way allows for
           multiple inputs.

            >>> image_embedding = Sequence(
            >>>     Input((3, 32, 32)), Flatten, Dense(640))
            >>> text_embedding = Sequence(Input((80,)), Embed(1000, 8), Flatten)
            >>> x = Sequence(Concat, Dense(64), Dense(1), Sigmoid)(
            >>>     image_embedding, text_embedding)

            >>> image_embedding = Input((3, 32, 32)) > Flatten > Dense(640) > Z
            >>> text_embedding = Input((80,)) > Embed(1000, 8) > Flatten > Z
            >>> x = Concat > Dense(64) > Dense(1) > Sigmoid > Z)(
            >>>     image_embedding, text_embedding)
            >>> x = (Conv(64) > Conv(64) > Conv(64) > Z)(image)

        4. Sequences being connected to previous layers in both ways is illegal
           and will summon the Internet Police.
        """
        super().__init__()
        assert nodes, 'Sequences cannot be empty.'
        nodes = list(map(_desugar, nodes))
        if in_nodes:
            # Connected to previous via graph __call__() syntax.
            assert not _is_input_or_has_in_nodes(nodes[0]), \
                'Tried to connect a sequence to previous layers via both ' + \
                'graph and sequence syntax (must choose one or the other).'
            self._in_nodes = []
            for node in in_nodes:
                self._in_nodes.append(node)
                node.add_out_node(self)
            start = 0
            self._model_input_nodes = _find_model_input_nodes(in_nodes)
        else:
            if _is_input_or_has_in_nodes(nodes[0]):
                # Connected to previous via first node of sequence.
                assert 2 <= len(nodes), 'Must have something to connect to.'
                self._in_nodes = [nodes[0]]
                nodes[0].add_out_node(self)
                start = 1
                self._model_input_nodes = _find_model_input_nodes([nodes[0]])
            else:
                # Orphan sequences.
                start = 0
                self._model_input_nodes = None
        specs = []
        for node in nodes[start:]:
            assert not _is_input_or_has_in_nodes(node)
            assert not node.out_nodes()
            specs.append(node.to_spec_or_specs())
        self._specs = specs
        self._is_model_built = False

    def __call__(self, *in_nodes):
        assert not self._in_nodes
        return Sequence(deepcopy(self._specs), in_nodes)

    def try_to_build(self):
        can_build, in_shapes, in_dtypes = self._gather_shapes_dtypes_for_build()
        if not can_build:
            return False
        self._layers, self._out_shape, self._out_dtype = \
            _build_recursive(self._specs, in_shapes, in_dtypes)
        for node in self._out_nodes:
            node.try_to_build()
        return True

    def is_built(self):
        return self._layers is not None

    def params(self):
        return _params_recursive(self._layers)

    def in_node_is_ready(self, is_training):
        assert self._in_nodes, \
            'Called in_node_is_ready() on a node with no inputs.'

        assert self._layers, \
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
        self._out_data = _forward_recursive(self._layers, xx, is_training)
        for node in self._out_nodes:
            node.in_node_is_ready(is_training)
        self._num_ready_in_nodes = 0

    def to_spec_or_specs(self):
        return self._specs

    def model_ensure_built(self):
        if self._is_model_built:
            return
        for node in self._model_input_nodes:
            assert node.try_to_build(), 'You are missing an input or inputs.'
        assert self._layers is not None, \
            'Your inputs and outputs are not all connected.'
        self._is_model_built = True

    def params_from(self, node, param_lists):
        id_ = id(node)
        if id_ in param_lists:
            return
        param_lists[id_] = node.params()
        for out_node in node.out_nodes():
            self.params_from(out_node, param_lists)

    def model_params(self):
        self.model_ensure_built()
        param_lists = {}
        for node in self._in_nodes:
            self.params_from(node, param_lists)
        ret = []
        for params in param_lists.values():
            ret += params
        return ret

    def model_forward(self, xx, is_training):
        self.model_ensure_built()
        assert len(xx) == len(self._model_input_nodes)
        for node, x in zip(self._in_nodes, xx):
            node.feed(x, is_training)
        assert self._out_data is not None
        return [self._out_data]
