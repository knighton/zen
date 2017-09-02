from .. import api as Z
from .optimizer import Optimizer


class MVO(Optimizer):
    """
    Minimum viable optimizer.
    """

    def __init__(self, lr=0.1):
        defaults = {'lr': lr}
        super().__init__(defaults)

    def step(self):
        for item in self.items:
            Z.update_grad(item.param, item.lr)
