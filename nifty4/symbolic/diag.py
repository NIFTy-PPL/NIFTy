from ..operators import DiagonalOperator
from .symbolic_tensor import SymbolicTensor


class SymbolicDiag(SymbolicTensor):
    def __init__(self, diag):
        assert diag.rank == 1
        self._diag = diag
        self._indices = (diag.indices[0], -diag.indices[0])

    def __call__(self, x):
        raise NotImplementedError

    def __str__(self):
        return 'diag({})'.format(self._diag)

    def eval(self, x):
        return DiagonalOperator(self._diag.eval(x))

    @property
    def derivative(self):
        raise NotImplementedError
