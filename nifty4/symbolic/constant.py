from .tensor import Tensor
from .symbolic_tensor import SymbolicTensor
from .zero import SymbolicZero


class SymbolicConstant(SymbolicTensor):
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
        return SymbolicZero(self.indices + (-1,), self._tensor.domain)
