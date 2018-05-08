from . import NLTensor
from ..operators import Tensor, ScalingOperator
from .constant import Constant


class Variable(NLTensor):
    def __init__(self, domain):
        self._domain = domain

    def __call__(self, x):
        raise ValueError

    def eval(self, x):
        return Tensor((1,), x)

    @property
    def derivative(self):
        return Constant(Tensor((1, -1), ScalingOperator(1., self._domain)))

