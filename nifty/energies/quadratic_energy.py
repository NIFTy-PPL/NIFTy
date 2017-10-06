from .energy import Energy
from ..memoization import memo


class QuadraticEnergy(Energy):
    """The Energy for a quadratic form.
    The most important aspect of this energy is that its curvature must be
    position-independent.
    """

    def __init__(self, position, A, b, gradient=None, curvature=None):
        super(QuadraticEnergy, self).__init__(position=position,
                                              gradient=gradient,
                                              curvature=curvature)
        self._A = A
        self._b = b
        if gradient is not None:
            self._Ax = gradient + self._b
        else:
            self._Ax = self._A(self.position)

    def at(self, position, gradient=None, curvature=None):
        return self.__class__(position=position, A=self._A, b=self._b,
                              gradient=gradient, curvature=curvature)

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
