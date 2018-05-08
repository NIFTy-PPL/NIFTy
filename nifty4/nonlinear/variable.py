from . import NLTensor
from ..operators import Tensor


class Variable(NLTensor):
    def __init__(self, domain):
        self._domain = domain

    def __call__(self, x):
        raise ValueError

    def eval(self, x):
        return Tensor((1,), x)

    @property
    def derivative(self):
        raise NotImplementedError
