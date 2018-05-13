from . import NLTensor
from ..operators import Tensor, ScalingOperator
from ..field import Field


class NLConstant(NLTensor):
    def __init__(self, tensor, indices):
        """
        Takes a tensor object and wraps it into a Nonlinear Object.
        """
        assert isinstance(tensor, Tensor)
        assert len(indices) == tensor.rank
        self._tensor = tensor
        self._indices = indices

    def __call__(self, x):
        return self

    def __str__(self):
        return self._tensor

    def eval(self, x):
        return self._tensor._thing

    @property
    def derivative(self):
        return NLZero(self.indices + (-1,), self._tensor.domain)


class NLZero(NLTensor):
    def __init__(self, indices, domain=None):
        self._indices = indices
        self._domain = domain

    def __call__(self, x):
        return self

    def __str__(self):
        return 'Zero'

    def eval(self, x):
        if self.rank == 2:
            return ScalingOperator(0, self._domain)
        elif self.rank == 1:
            return Field.zeros(self._domain)
        elif self.rank == 0:
            return Field.zeros(())
        else:
            # FIXME This situation here is suboptimal
            return 0

    @property
    def derivative(self):
        return NLZero(self.indices + (-1,))
