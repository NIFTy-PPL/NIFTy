from ..minimization.energy import Energy
from ..operators.inversion_enabler import InversionEnabler
from .wiener_filter_curvature import WienerFilterCurvature


class WienerFilterEnergy(Energy):
    """The Energy for the Wiener filter.

    It covers the case of linear measurement with
    Gaussian noise and Gaussian signal prior with known covariance.

    Parameters
    ----------
    position: Field,
        The current position.
    d: Field,
       the data
    R: LinearOperator,
       The response operator, description of the measurement process.
    N: EndomorphicOperator,
       The noise covariance in data space.
    S: EndomorphicOperator,
       The prior signal covariance in harmonic space.
    """

    def __init__(self, position, d, R, N, S, inverter, _j=None):
        super(WienerFilterEnergy, self).__init__(position=position)
        self.R = R
        self.N = N
        self.S = S
        self._curvature = InversionEnabler(WienerFilterCurvature(R, N, S),
                                           inverter=inverter)
        self._inverter = inverter
        if _j is None:
            _j = self.R.adjoint_times(self.N.inverse_times(d))
        self._j = _j
        Dx = self._curvature(self.position)
        self._value = 0.5*self.position.vdot(Dx) - self._j.vdot(self.position)
        self._gradient = Dx - self._j

    def at(self, position):
        return self.__class__(position=position, d=None, R=self.R, N=self.N,
                              S=self.S, inverter=self._inverter, _j=self._j)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @property
    def curvature(self):
        return self._curvature
