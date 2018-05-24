from .symbolic_tensor import SymbolicTensor


class SymbolicAdjoint(SymbolicTensor):
    def __init__(self, thing, indices=None):
        # MR FIXME: do we actually use the "indices" argument?
        assert thing.rank in [1, 2]
        if indices is None:
            if thing.rank == 1:
                indices = (-1 * thing.indices[0],)
            else:
                indices = thing.indices[::-1]
        super(SymbolicAdjoint, self).__init__(indices)
        self._thing = thing


    @staticmethod
    def make(thing):
        if isinstance(thing, SymbolicAdjoint):
            return thing._thing
        return SymbolicAdjoint(thing)

    def __str__(self):
        return '{}^dagger'.format(self._thing)

    def eval(self, x):
        res = self._thing.eval(x)
        if isinstance(res, float) and res == 0.:
            return 0.
        if self.rank == 2:
            return res.adjoint
        elif self.rank == 1:
            return res.conjugate()
        else:
            raise NotImplementedError

    @property
    def derivative(self):
        if self.rank == 1:
            return self.__class__(self._thing.derivative, self._indices + (-1,))
        raise NotImplementedError
