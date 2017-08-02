from nifty.energies.energy import Energy
from nifty.energies.memoization import memo
from nifty.library.log_normal_wiener_filter import \
    LogNormalWienerFilterCurvature
from nifty.sugar import create_composed_fft_operator


class LogNormalWienerFilterEnergy(Energy):
    """The Energy for the log-normal Wiener filter.

    It covers the case of linear measurement with
    Gaussian noise and Gaussain signal prior with known covariance.

    Parameters
    ----------
    position: Field,
        The current position.
    d : Field,
        the data.
    R : Operator,
        The response operator, describtion of the measurement process.
    N : EndomorphicOperator,
        The noise covariance in data space.
    S : EndomorphicOperator,
        The prior signal covariance in harmonic space.
    """

    def __init__(self, position, d, R, N, S, fft4exp=None):
        super(LogNormalWienerFilterEnergy, self).__init__(position=position)
        self.d = d
        self.R = R
        self.N = N
        self.S = S

        if fft4exp is None:
            self._fft = create_composed_fft_operator(self.S.domain,
                                                     all_to='position')
        else:
            self._fft = fft4exp

    def at(self, position):
        return self.__class__(position=position, d=self.d, R=self.R, N=self.N,
                              S=self.S, fft4exp=self._fft)

    @property
    @memo
    def value(self):
        return 0.5*(self.position.vdot(self._Sp) +
                    self._Rexppd.vdot(self._NRexppd))

    @property
    @memo
    def gradient(self):
        return self._Sp + self._exppRNRexppd

    @property
    @memo
    def curvature(self):
        return LogNormalWienerFilterCurvature(R=self.R, N=self.N, S=self.S,
                                              d=self.d, position=self.position,
                                              fft4exp=self._fft)

    @property
    def _expp(self):
        return self.curvature._expp

    @property
    def _Rexppd(self):
        return self.curvature._Rexppd

    @property
    def _NRexppd(self):
        return self.curvature._NRexppd

    @property
    def _exppRNRexppd(self):
        return self.curvature._exppRNRexppd

    @property
    @memo
    def _Sp(self):
        return self.S.inverse_times(self.position)
