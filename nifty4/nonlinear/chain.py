from .tensor import NLTensor


class NLChain(NLTensor):
    def __init__(self, outer, inner):
        assert outer.rank == 1 and inner.rank == 1
        self._outer = outer
        self._inner = inner
        self._indices = outer.indices

    def __str__(self):
        return '{}({})'.format(self._outer, self._inner)

    def eval(self, x):
        return self._outer.eval(self._inner.eval(x))

    @property
    def derivative(self):
        return self.__class__(self._outer, self._inner.derivative)
