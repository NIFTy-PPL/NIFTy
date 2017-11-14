from .energy import Energy
from ..utilities import memo


class QuadraticEnergy(Energy):
    """The Energy for a quadratic form.
    The most important aspect of this energy is that its curvature must be
    position-independent.
    """

    def __init__(self, position, A, b, _grad=None):
        super(QuadraticEnergy, self).__init__(position=position)
        self._A = A
        self._b = b
        if _grad is not None:
            self._Ax = _grad + self._b
        else:
            self._Ax = self._A(self.position)

    def at(self, position):
        return QuadraticEnergy(position=position, A=self._A, b=self._b)

    def at_with_grad(self, position, grad):
        return QuadraticEnergy(position=position, A=self._A, b=self._b,
                               _grad=grad)

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
