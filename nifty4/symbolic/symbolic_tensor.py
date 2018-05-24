class SymbolicTensor(object):
    def __init__(self, indices):
        self._indices = indices

    def __call__(self, x):
        from .contractions import SymbolicChain
        return SymbolicChain(self, x)

    @property
    def indices(self):
        return self._indices

    @property
    def rank(self):
        return len(self._indices)

    def eval(self, x):
        raise NotImplementedError

    @property
    def derivative(self):
        raise NotImplementedError

    @property
    def adjoint(self):
        if self.rank in [1, 2]:
            from .adjoint import SymbolicAdjoint
            return SymbolicAdjoint.make(self)
        raise NotImplementedError
