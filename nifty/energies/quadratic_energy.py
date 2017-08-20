from nifty.energies.energy import Energy
from nifty.energies.memoization import memo


class QuadraticEnergy(Energy):
    """The Energy for a quadratic form.
    The most important aspect of this energy is that its curvature must be
    position-independent.
    """

    def __init__(self, position, A, b, grad=None):
        super(QuadraticEnergy, self).__init__(position=position)
        self._A = A
        self._b = b
        if grad is not None:
            self._Ax = grad + self._b
        else:
            self._Ax = self._A(self.position)

    def at(self, position):
        return self.__class__(position=position, A=self._A, b=self._b)

    def at_with_grad(self, position, grad):
        return self.__class__(position=position, A=self._A, b=self._b,
                              grad=grad)

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
