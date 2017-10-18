from ...energies.energy import Energy
from ...memoization import memo
from ...minimization import ConjugateGradient
from . import LogNormalWienerFilterCurvature
from ...sugar import create_composed_fft_operator


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

    def __init__(self, position, d, R, N, S, offset=None, fft4exp=None,
                 inverter=None, gradient=None, curvature=None):
        super(LogNormalWienerFilterEnergy, self).__init__(position=position,
                                                          gradient=gradient,
                                                          curvature=curvature)
        self.d = d
        self.R = R
        self.N = N
        self.S = S
        self.offset = offset
        if fft4exp is None:
            self._fft = create_composed_fft_operator(self.S.domain,
                                                     all_to='position')
        else:
            self._fft = fft4exp

        if inverter is None:
            inverter = ConjugateGradient(preconditioner=self.S.times)
        self._inverter = inverter

    @property
    def inverter(self):
        return self._inverter

    def at(self, position, gradient=None, curvature=None):
        return self.__class__(position=position,
                              d=self.d, R=self.R, N=self.N, S=self.S,
                              offset=self.offset, fft4exp=self._fft,
                              gradient=gradient, curvature=curvature)

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
        return LogNormalWienerFilterCurvature(R=self.R,
                                              N=self.N,
                                              S=self.S,
                                              d=self.d,
                                              position=self.position,
                                              fft4exp=self._fft,
                                              offset=self.offset,
                                              inverter=self.inverter)

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
