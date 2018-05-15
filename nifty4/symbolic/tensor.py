class NLTensor(object):
    def __call__(self, x):
        from .contractions import NLChain
        return NLChain(self, x)

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
            from .adjoint import NLAdjoint
            return NLAdjoint(self)
        else:
            raise NotImplementedError
