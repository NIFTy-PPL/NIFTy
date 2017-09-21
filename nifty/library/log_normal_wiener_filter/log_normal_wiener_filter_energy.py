from ...energies.energy import Energy
from ...energies.memoization import memo
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

    def __init__(self, position, d, R, N, S, fft4exp=None, old_curvature=None,
                 offset=None):
        super(LogNormalWienerFilterEnergy, self).__init__(position=position)
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

        self._old_curvature = old_curvature
        self._curvature = None

    def at(self, position):
        return self.__class__(position=position, d=self.d, R=self.R, N=self.N,
                              S=self.S, fft4exp=self._fft,
                              old_curvature=self._curvature,
                              offset=self.offset)

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
    def curvature(self):
        if self._curvature is None:
            if self._old_curvature is None:
                self._curvature = LogNormalWienerFilterCurvature(
                                                      R=self.R,
                                                      N=self.N,
                                                      S=self.S,
                                                      d=self.d,
                                                      position=self.position,
                                                      fft4exp=self._fft,
                                                      offset=self.offset)
            else:
                self._curvature = \
                    self._old_curvature.copy(position=self.position)
        return self._curvature

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
