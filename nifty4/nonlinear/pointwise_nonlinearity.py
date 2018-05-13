from . import NLTensor, NLCABF
from ..field import exp


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
        raise NotImplementedError
        # return NLCABF(NLDiag(NLExp(self._inner), -1), self._inner.derivative)
