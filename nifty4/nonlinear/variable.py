from . import NLTensor
from ..operators import Tensor, ScalingOperator
from .constant import NLConstant


class NLVariable(NLTensor):
    def __init__(self, domain):
        self._domain = domain

    def __call__(self, x):
        raise ValueError

    def __str__(self):
        return 'var'

    def eval(self, x):
        return Tensor((1,), x)

    @property
    def derivative(self):
        return NLConstant(Tensor((1, -1), ScalingOperator(1., self._domain)))

