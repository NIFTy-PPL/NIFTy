from ..sugar import exp, tanh
from .contractions import SymbolicChainLinOps
from .diag import SymbolicDiag
from .symbolic_tensor import SymbolicTensor


class PointwiseNonlinearity(SymbolicTensor):
    def __init__(self, inner):
        assert inner.rank == 1
        super(PointwiseNonlinearity, self).__init__(inner.indices)
        self._inner = inner

    def __call__(self, x):
        raise NotImplementedError


class SymbolicNonlinear(PointwiseNonlinearity):
    def __init__(self, name, func, inner, deriv=None):
        super(SymbolicNonlinear, self).__init__(inner)
        self._name = name
        self._func = func
        self._deriv = deriv

    def __str__(self):
        return '{}({})'.format(self._name, self._inner)

    def eval(self, x):
        return self._func(self._inner.eval(x))

    @property
    def derivative(self):
        if self._deriv is None:
            return NotImplementedError
        tmp = SymbolicNonlinear(self._name+"'", self._deriv, self._inner)
        return SymbolicChainLinOps(SymbolicDiag(tmp), self._inner.derivative)


def fromNiftyNL(name, NL, inner):
    return SymbolicNonlinear(name, NL.__call__, inner, NL.derivative)
