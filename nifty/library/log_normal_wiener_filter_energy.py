from ..minimization.energy import Energy
from ..utilities import memo
from .log_normal_wiener_filter_curvature import LogNormalWienerFilterCurvature
from ..sugar import create_composed_fft_operator
from ..field import exp


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

    def __init__(self, position, d, R, N, S, inverter, fft=None):
        super(LogNormalWienerFilterEnergy, self).__init__(position=position)
        self.d = d
        self.R = R
        self.N = N
        self.S = S
        self._inverter = inverter

        if fft is None:
            self._fft = create_composed_fft_operator(self.S.domain,
                                                     all_to='position')
        else:
            self._fft = fft

        self._expp_sspace = exp(self._fft(self.position))

        Sp = self.S.inverse_times(self.position)
        expp = self._fft.adjoint_times(self._expp_sspace)
        Rexppd = self.R(expp) - self.d
        NRexppd = self.N.inverse_times(Rexppd)
        self._value = 0.5*(self.position.vdot(Sp) + Rexppd.vdot(NRexppd))
        exppRNRexppd = self._fft.adjoint_times(
            self._expp_sspace * self._fft(self.R.adjoint_times(NRexppd)))
        self._gradient = Sp + exppRNRexppd

    def at(self, position):
        return self.__class__(position, self.d, self.R, self.N, self.S,
                              self._inverter, self._fft)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @property
    @memo
    def curvature(self):
        return LogNormalWienerFilterCurvature(
            R=self.R, N=self.N, S=self.S, fft=self._fft,
            expp_sspace=self._expp_sspace, inverter=self._inverter)
