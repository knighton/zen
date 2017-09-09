from .model import Model


class Graph(Model):
    def __init__(self, in_nodes, out_nodes):
        if not hasattr(in_nodes, '__getitem__'):
            in_nodes = [in_nodes]
        if not hasattr(out_nodes, '__getitem__'):
            out_nodes = [out_nodes]
        self._in_nodes = in_nodes
        self._out_nodes = out_nodes
        self._is_built = False

    def ensure_built(self):
        if self._is_built:
            return
        for node in self._in_nodes:
            assert node.try_to_build(), 'You are missing an input or inputs.'
        for node in self._out_nodes:
            assert node.is_built(), \
                'Your inputs and outputs are not all connected.'
        self._is_built = True

    def params_from(self, node, param_lists):
        id_ = id(node)
        if id_ in param_lists:
            return
        param_lists[id_] = node.params()
        for out_node in node.out_nodes():
            self.params_from(out_node, param_lists)

    def model_params(self):
        self.ensure_built()
        param_lists = {}
        for node in self._in_nodes:
            self.params_from(node, param_lists)
        ret = []
        for params in param_lists.values():
            ret += params
        return ret

    def model_forward(self, xx, is_training):
        self.ensure_built()
        assert len(xx) == len(self._in_nodes)
        for node, x in zip(self._in_nodes, xx):
            node.feed(x, is_training)
        yy = []
        for node in self._out_nodes:
            y = node.out_data()
            assert y is not None
            yy.append(y)
        return yy
