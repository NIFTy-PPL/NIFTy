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


class SymbolicLinear(PointwiseNonlinearity):
    def __str__(self):
        return 'linear({})'.format(self._inner)

    def eval(self, x):
        return self._inner.eval(x)

    @property
    def derivative(self):
        return self._inner.derivative


class SymbolicExp(PointwiseNonlinearity):
    def __str__(self):
        return 'exp({})'.format(self._inner)

    def eval(self, x):
        return exp(self._inner.eval(x))

    @property
    def derivative(self):
        return SymbolicChainLinOps(SymbolicDiag(SymbolicExpPrime(self._inner)),
                             self._inner.derivative)


class SymbolicExpPrime(PointwiseNonlinearity):
    def __str__(self):
        return "exp'({})".format(self._inner)

    def eval(self, x):
        return exp(self._inner.eval(x))

    @property
    def derivative(self):
        # FIXME This function should be triggered in the demo but it isn't.
        # As soon as it is replace the error with a zero in order to implement
        # Jakob's curvature
        raise NotImplementedError


class SymbolicTanh(PointwiseNonlinearity):
    def __str__(self):
        return 'tanh({})'.format(self._inner)

    def eval(self, x):
        return tanh(self._inner.eval(x))

    @property
    def derivative(self):
        return SymbolicChainLinOps(SymbolicDiag(SymbolicTanhPrime(self._inner)),
                             self._inner.derivative)


class SymbolicTanhPrime(PointwiseNonlinearity):
    def __str__(self):
        return "tanh'({})".format(self._inner)

    def eval(self, x):
        return 1 - tanh(self._inner.eval(x))**2

    @property
    def derivative(self):
        # FIXME This function should be triggered in the demo but it isn't.
        # As soon as it is replace the error with a zero in order to implement
        # Jakob's curvature
        raise NotImplementedError
