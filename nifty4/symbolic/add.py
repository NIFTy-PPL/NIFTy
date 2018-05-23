from .symbolic_tensor import SymbolicTensor


class SymbolicAdd(SymbolicTensor):
    def __init__(self, fst, snd):
        assert fst.indices == snd.indices
        super(SymbolicAdd, self).__init__(fst.indices)
        self._fst = fst
        self._snd = snd

    def __str__(self):
        return '{} + {}'.format(self._fst, self._snd)

    def __call__(self, x):
        return self._fst(x) + self._snd(x)

    def eval(self, x):
        A = self._fst.eval(x)
        B = self._snd.eval(x)
        if isinstance(A, float) and A == 0.:
            return B
        if isinstance(B, float) and B == 0.:
            return A
        return A + B

    @property
    def derivative(self):
        return self.__class__(self._fst.derivative,  self._snd.derivative)

    @property
    def curvature(self):
        return self.__class__(self._fst.curvature,  self._snd.curvature)
