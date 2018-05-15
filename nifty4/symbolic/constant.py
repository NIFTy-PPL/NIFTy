from ..operators import Tensor
from .tensor import NLTensor
from .zero import NLZero


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
        return str(self._tensor)

    def eval(self, x):
        return self._tensor._thing

    @property
    def derivative(self):
        return NLZero(self.indices + (-1,), self._tensor.domain)
