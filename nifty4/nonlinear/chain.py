from . import NLTensor


class Chain(NLTensor):
    def __init__(self, inner, outer):
        # FIXME Check indices
        self._outer = outer
        self._inner = inner

    def eval(self, x):
        return self._outer.eval(self._inner.eval(x))

    @property
    def derivative(self):
        pass
