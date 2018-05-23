from .symbolic_tensor import SymbolicTensor


class SymbolicZero(SymbolicTensor):
    def __init__(self, indices, domain=None):
        super(SymbolicZero, self).__init__(indices)
        self._domain = domain

    def __call__(self, x):
        return self

    def __str__(self):
        return 'Zero'

    def eval(self, x):
        return 0.
        # FIXME This situation here is suboptimal. It would be much better to
        # return the actual NIFTy objects.

    @property
    def derivative(self):
        return self.__class__(self.indices + (-1,))
