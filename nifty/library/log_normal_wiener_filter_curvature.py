from ..operators import EndomorphicOperator, InversionEnabler
from ..utilities import memo
from ..field import exp


class LogNormalWienerFilterCurvature(EndomorphicOperator):
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

    class _Helper(EndomorphicOperator):
        def __init__(self, R, N, S, position, fft4exp):
            super(LogNormalWienerFilterCurvature._Helper, self).__init__()
            self.R = R
            self.N = N
            self.S = S
            self.position = position
            self._fft = fft4exp

        @property
        def domain(self):
            return self.S.domain

        @property
        def capability(self):
            return self.TIMES | self.ADJOINT_TIMES

        def apply(self, x, mode):
            self._check_input(x, mode)
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

    def __init__(self, R, N, S, position, fft4exp, inverter):
        super(LogNormalWienerFilterCurvature, self).__init__()
        self._op = self._Helper(R, N, S, position, fft4exp)
        self._op = InversionEnabler(self._op, inverter)

    @property
    def domain(self):
        return self._op.domain

    @property
    def capability(self):
        return self._op.capability

    @property
    def _expp_sspace(self):
        return self._op._op._expp_sspace

    def apply(self, x, mode):
        return self._op.apply(x, mode)
