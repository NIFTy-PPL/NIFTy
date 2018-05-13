from . import NLTensor
from ..operators import Tensor


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
        return NLZero(self.indices + (-1,))


class NLZero(NLTensor):
    def __init__(self, indices):
        self._indices = indices

    def __call__(self, x):
        return self

    def __str__(self):
        return 'Zero'

    def eval(self, x):
        return 0

    @property
    def derivative(self):
        return NLZero(self.indices + (-1,))
