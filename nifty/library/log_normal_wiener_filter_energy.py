from ..minimization.energy import Energy
from ..utilities import memo
from .log_normal_wiener_filter_curvature import LogNormalWienerFilterCurvature
from ..sugar import create_composed_fft_operator


class LogNormalWienerFilterEnergy(Energy):
    """The Energy for the log-normal Wiener filter.

    It covers the case of linear measurement with
    Gaussian noise and Gaussain signal prior with known covariance.

    Parameters
    ----------
    position: Field,
       The current position.
    d: Field,
       the data.
    R: Operator,
       The response operator, describtion of the measurement process.
    N: EndomorphicOperator,
       The noise covariance in data space.
    S: EndomorphicOperator,
       The prior signal covariance in harmonic space.
    """

    def __init__(self, position, d, R, N, S, inverter, fft4exp=None):
        super(LogNormalWienerFilterEnergy, self).__init__(position=position)
        self.d = d
        self.R = R
        self.N = N
        self.S = S
        self._inverter = inverter

        if fft4exp is None:
            self._fft = create_composed_fft_operator(self.S.domain,
                                                     all_to='position')
        else:
            self._fft = fft4exp

    def at(self, position):
        return self.__class__(position=position, d=self.d, R=self.R, N=self.N,
                              S=self.S, fft4exp=self._fft,
                              inverter=self._inverter)

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
        return LogNormalWienerFilterCurvature(
            R=self.R, N=self.N, S=self.S, position=self.position,
            fft4exp=self._fft, inverter=self._inverter)

    @property
    @memo
    def _Sp(self):
        return self.S.inverse_times(self.position)

    @property
    @memo
    def _Rexppd(self):
        expp = self._fft.adjoint_times(self.curvature._expp_sspace)
        return self.R(expp) - self.d

    @property
    @memo
    def _NRexppd(self):
        return self.N.inverse_times(self._Rexppd)

    @property
    @memo
    def _exppRNRexppd(self):
        return self._fft.adjoint_times(
                    self.curvature._expp_sspace *
                    self._fft(self.R.adjoint_times(self._NRexppd)))
