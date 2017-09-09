from copy import deepcopy

from ..arch.vee import Vee
from .node import LayerNode
from .spec import Spec


class Sugar(Vee):
    """
    A spec factory with default arguments.  Returns orphan layer nodes.
    """

    def __init__(self, spec_class, default_kwargs=None):
        assert issubclass(spec_class, Spec)
        if default_kwargs is None:
            default_kwargs = {}
        super().__init__()
        self.spec_class = spec_class
        self.default_kwargs = default_kwargs

    def __call__(self, *args, **override_kwargs):
        kwargs = deepcopy(self.default_kwargs)
        kwargs.update(deepcopy(override_kwargs))
        spec = self.spec_class(*args, **kwargs)
        return LayerNode(spec)
