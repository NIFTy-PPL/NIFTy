from .energy import Energy


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
            self._grad = _grad
            Ax = _grad + self._b
        else:
            Ax = self._A(self.position)
            self._grad = Ax - self._b
        self._value = 0.5*self.position.vdot(Ax) - b.vdot(self.position)

    def at(self, position):
        return QuadraticEnergy(position=position, A=self._A, b=self._b)

    def at_with_grad(self, position, grad):
        return QuadraticEnergy(position=position, A=self._A, b=self._b,
                               _grad=grad)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._grad

    @property
    def curvature(self):
        return self._A
