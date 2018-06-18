from ..minimization import Energy
from ..operators import InversionEnabler, SandwichOperator
from ..utilities import memo


class UnitLogGauss(Energy):
    def __init__(self, position, s, inverter=None):
        """
        s: Sky model object

        value = 0.5 * s.vdot(s), i.e. a log-Gauss distribution with unit
        covariance
        """
        super(UnitLogGauss, self).__init__(position)
        self._s = s.at(position)
        self._inverter = inverter

    def at(self, position):
        return self.__class__(position, self._s, self._inverter)

    @property
    @memo
    def _gradient_helper(self):
        return self._s.gradient

    @property
    @memo
    def value(self):
        return .5 * self._s.value.vdot(self._s.value)

    @property
    @memo
    def gradient(self):
        return self._gradient_helper.adjoint(self._s.value)

    @property
    @memo
    def curvature(self):
        c = SandwichOperator.make(self._gradient_helper)
        if self._inverter is None:
            return c
        return InversionEnabler(c, self._inverter)
