from . import NLTensor, NLChainLinOps
from ..field import exp
from ..operators import DiagonalOperator


class NLExp(NLTensor):
    def __init__(self, inner):
        assert inner.rank == 1
        self._inner = inner
        self._indices = inner.indices

    def __call__(self, x):
        raise NotImplementedError

    def __str__(self):
        return 'exp({})'.format(self._inner)

    def eval(self, x):
        return exp(self._inner.eval(x))

    @property
    def derivative(self):
        return NLChainLinOps(NLDiag(NLExp(self._inner)), self._inner.derivative)


class NLDiag(NLTensor):
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
