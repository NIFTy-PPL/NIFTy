from .energy import Energy
from .memoization import memo


class QuadraticEnergy(Energy):
    """The Energy for a quadratic form.
    The most important aspect of this energy is that its curvature must be
    position-independent.
    """

    def __init__(self, position, A, b, _grad=None, _bnorm=None):
        super(QuadraticEnergy, self).__init__(position=position)
        self._A = A
        self._b = b
        self._bnorm = _bnorm
        if _grad is not None:
            self._Ax = _grad + self._b
        else:
            self._Ax = self._A(self.position)

    def at(self, position):
        return self.__class__(position=position, A=self._A, b=self._b,
                              _bnorm=self.norm_b)

    def at_with_grad(self, position, grad):
        return self.__class__(position=position, A=self._A, b=self._b,
                              _grad=grad, _bnorm=self.norm_b)

    @property
    @memo
    def value(self):
        return 0.5*self.position.vdot(self._Ax) - self._b.vdot(self.position)

    @property
    @memo
    def gradient(self):
        return self._Ax - self._b

    @property
    def curvature(self):
        return self._A

    @property
    def norm_b(self):
        if self._bnorm is None:
            self._bnorm = self._b.norm()
        return self._bnorm
