from .symbolic_tensor import SymbolicTensor


class SymbolicAdjoint(SymbolicTensor):
    def __init__(self, thing, indices=None):
        assert thing.rank in [1, 2]
        if indices is None:
            if thing.rank == 1:
                self._indices = (-1 * thing.indices[0],)
            else:
                self._indices = thing.indices[::-1]
        else:
            self._indices = indices
        self._thing = thing

    def __str__(self):
        return '{}^dagger'.format(self._thing)

    def eval(self, x):
        if isinstance(self._thing.eval(x), float) and self._thing.eval(x) == 0.:
            return 0.
        if self.rank == 2:
            return self._thing.eval(x).adjoint
        elif self.rank == 1:
            return self._thing.eval(x).conjugate()
        else:
            raise NotImplementedError

    @property
    def derivative(self):
        if self.rank == 1:
            return self.__class__(self._thing.derivative, self._indices + (-1,))
        raise NotImplementedError
