from ..field import exp, tanh
from .contractions import NLChainLinOps
from .diag import NLDiag
from .tensor import NLTensor


class PointwiseNonlinearity(NLTensor):
    def __init__(self, inner):
        assert inner.rank == 1
        self._inner = inner
        self._indices = inner.indices

    def __call__(self, x):
        raise NotImplementedError


class NLLinear(PointwiseNonlinearity):
    def __str__(self):
        return 'linear({})'.format(self._inner)

    def eval(self, x):
        return self._inner.eval(x)

    @property
    def derivative(self):
        return self._inner.derivative


class NLExp(PointwiseNonlinearity):
    def __str__(self):
        return 'exp({})'.format(self._inner)

    def eval(self, x):
        return exp(self._inner.eval(x))

    @property
    def derivative(self):
        return NLChainLinOps(NLDiag(NLExp(self._inner)),
                             self._inner.derivative)


class NLTanh(PointwiseNonlinearity):
    def __str__(self):
        return 'tanh({})'.format(self._inner)

    def eval(self, x):
        return tanh(self._inner.eval(x))

    @property
    def derivative(self):
        return NLChainLinOps(NLDiag(NLTanhPrime(self._inner)),
                             self._inner.derivative)


class NLTanhPrime(PointwiseNonlinearity):
    def __str__(self):
        return "tanh'({})".format(self._inner)

    def eval(self, x):
        return 1 - tanh(self._inner.eval(x))**2

    @property
    def derivative(self):
        raise NotImplementedError
