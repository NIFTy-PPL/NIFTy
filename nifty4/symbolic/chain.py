from .symbolic_tensor import SymbolicTensor


class SymbolicChain(SymbolicTensor):
    def __init__(self, outer, inner):
        assert outer.rank == 1 and inner.rank == 1
        super(SymbolicChain, self).__init__(outer.indices)
        self._outer = outer
        self._inner = inner

    def __str__(self):
        return '{}({})'.format(self._outer, self._inner)

    def eval(self, x):
        return self._outer.eval(self._inner.eval(x))

    @property
    def derivative(self):
        return self.__class__(self._outer, self._inner.derivative)
