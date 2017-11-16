from ..operators import EndomorphicOperator
from ..utilities import memo
from ..field import exp
from ..sugar import create_composed_fft_operator


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

    def __init__(self, R, N, S, d, position, fft4exp=None):
        self.R = R
        self.N = N
        self.S = S
        self.d = d
        self.position = position

        if fft4exp is None:
            self._fft = create_composed_fft_operator(self.domain,
                                                     all_to='position')
        else:
            self._fft = fft4exp

        super(LogNormalWienerFilterCurvature, self).__init__()

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
        # part2 = self._exppRNRexppd * x
        part3 = self._fft.adjoint_times(self._expp_sspace * self._fft(x))
        part3 = self._fft.adjoint_times(
                    self._expp_sspace *
                    self._fft(self.R.adjoint_times(
                                self.N.inverse_times(self.R(part3)))))
        return part1 + part3  # + part2

    @property
    @memo
    def _expp_sspace(self):
        return exp(self._fft(self.position))

    @property
    @memo
    def _Rexppd(self):
        expp = self._fft.adjoint_times(self._expp_sspace)
        return self.R(expp) - self.d

    @property
    @memo
    def _NRexppd(self):
        return self.N.inverse_times(self._Rexppd)

    @property
    @memo
    def _exppRNRexppd(self):
        return self._fft.adjoint_times(
                    self._expp_sspace *
                    self._fft(self.R.adjoint_times(self._NRexppd)))
