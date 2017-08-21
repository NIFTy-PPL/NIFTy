from nifty.operators import EndomorphicOperator,\
                            InvertibleOperatorMixin
from nifty.energies.memoization import memo
from nifty.basic_arithmetics import clipped_exp
from nifty.sugar import create_composed_fft_operator


class LogNormalWienerFilterCurvature(InvertibleOperatorMixin,
                                     EndomorphicOperator):
    """The curvature of the LogNormalWienerFilterEnergy.

    This operator implements the second derivative of the
    LogNormalWienerFilterEnergy used in some minimization algorithms or for
    error estimates of the posterior maps. It is the inverse of the propagator
    operator.

    Parameters
    ----------
    R: LinearOperator,
        The response operator of the Wiener filter measurement.
    N : EndomorphicOperator
        The noise covariance.
    S: DiagonalOperator,
        The prior signal covariance

    """

    def __init__(self, R, N, S, d, position, inverter=None,
                 preconditioner=None, fft4exp=None, **kwargs):
        self._cache = {}
        self.R = R
        self.N = N
        self.S = S
        self.d = d
        self.position = position
        if preconditioner is None:
            preconditioner = self.S.times
        self._domain = self.S.domain

        if fft4exp is None:
            self._fft = create_composed_fft_operator(self.domain,
                                                     all_to='position')
        else:
            self._fft = fft4exp

        super(LogNormalWienerFilterCurvature, self).__init__(
                                                 inverter=inverter,
                                                 preconditioner=preconditioner,
                                                 **kwargs)

    @property
    def domain(self):
        return self._domain

    @property
    def self_adjoint(self):
        return True

    @property
    def unitary(self):
        return False

    # ---Added properties and methods---

    def _times(self, x, spaces):
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
        return clipped_exp(self._fft(self.position))

    @property
    @memo
    def _expp(self):
        return self._fft.adjoint_times(self._expp_sspace)

    @property
    @memo
    def _Rexppd(self):
        return self.R(self._expp) - self.d

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
