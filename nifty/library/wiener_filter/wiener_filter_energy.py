from ...energies.energy import Energy
from ...utilities import memo
from ...operators.inversion_enabler import InversionEnabler
from . import WienerFilterCurvature


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
        self.d = d
        self.R = R
        self.N = N
        self.S = S
        self._inverter = inverter
        self._jpre = _j

    def at(self, position):
        return self.__class__(position=position, d=self.d, R=self.R, N=self.N,
                              S=self.S, inverter=self._inverter, _j=self._jpre)

    @property
    @memo
    def value(self):
        return 0.5*self.position.vdot(self._Dx) - self._j.vdot(self.position)

    @property
    @memo
    def gradient(self):
        return self._Dx - self._j

    @property
    @memo
    def curvature(self):
        return InversionEnabler(WienerFilterCurvature(R=self.R, N=self.N,
                                                      S=self.S),
                                inverter=self._inverter)

    @property
    @memo
    def _Dx(self):
        return self.curvature(self.position)

    @property
    def _j(self):
        if self._jpre is None:
            self._jpre = self.R.adjoint_times(self.N.inverse_times(self.d))
        return self._jpre
