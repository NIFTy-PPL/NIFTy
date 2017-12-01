from ..operators import EndomorphicOperator, InversionEnabler
from ..utilities import memo
from ..field import exp


class LogNormalWienerFilterCurvature(InversionEnabler, EndomorphicOperator):
    """The curvature of the LogNormalWienerFilterEnergy.

    This operator implements the second derivative of the
    LogNormalWienerFilterEnergy used in some minimization algorithms or for
    error estimates of the posterior maps. It is the inverse of the propagator
    operator.

    Parameters
    ----------
    R: LinearOperator,
       The response operator of the Wiener filter measurement.
    N: EndomorphicOperator
       The noise covariance.
    S: DiagonalOperator,
       The prior signal covariance
    """

    def __init__(self, R, N, S, position, fft4exp, inverter):
        InversionEnabler.__init__(self, inverter)
        EndomorphicOperator.__init__(self)
        self.R = R
        self.N = N
        self.S = S
        self.position = position
        self._fft = fft4exp

    @property
    def domain(self):
        return self.S.domain

    @property
    def self_adjoint(self):
        return True

    @property
    def unitary(self):
        return False

    def _times(self, x):
        part1 = self.S.inverse_times(x)
        part3 = self._fft.adjoint_times(self._expp_sspace * self._fft(x))
        part3 = self._fft.adjoint_times(
                    self._expp_sspace *
                    self._fft(self.R.adjoint_times(
                                self.N.inverse_times(self.R(part3)))))
        return part1 + part3

    @property
    @memo
    def _expp_sspace(self):
        return exp(self._fft(self.position))
