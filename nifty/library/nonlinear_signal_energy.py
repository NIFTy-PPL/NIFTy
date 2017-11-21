from ...energies.energy import Energy
from ...memoization import memo
from .nonlinear_signal_curvature import NonlinearSignalCurvature
from .response_operators import LinearizedSignalResponse


class NonlinearWienerFilterEnergy(Energy):
    """The Energy for the Gaussian lognormal case.

    It describes the situation of linear measurement  of a
    lognormal signal with Gaussian noise and Gaussain signal prior.

    Parameters
    ----------
    d : Field,
        the data.
    R : Operator,
        The nonlinear response operator, describtion of the measurement process.
    N : EndomorphicOperator,
        The noise covariance in data space.
    S : EndomorphicOperator,
        The prior signal covariance in harmonic space.
    """

    def __init__(self, position, d, Instrument, nonlinearity, FFT, power, N, S, inverter=None):
        super(NonlinearWienerFilterEnergy, self).__init__(position=position)
        # print "init", position.norm()
        self.d = d
        self.Instrument = Instrument
        self.nonlinearity = nonlinearity
        self.FFT = FFT
        self.power = power
        self.LinearizedResponse = LinearizedSignalResponse(Instrument, nonlinearity,
                                                           FFT, power, self.position)

        position_map = FFT.adjoint_times(self.power * self.position)
        # position_map = (Field(FFT.domain,val=position_map.val.real+0j))
        self.residual = d - Instrument(nonlinearity(position_map))
        # Field(d.domain,
        # val=self.residual.val.get_full_data().view(np.complex128).conjugate().view(np.float64))
        self.N = N
        self.S = S
        self.inverter = inverter
        self._t1 = self.S.inverse_times(self.position)
        self._t2 = self.N.inverse_times(self.residual)

    def at(self, position):
        return self.__class__(position, self.d, self.Instrument,
                              self.nonlinearity, self.FFT, self.power, self.N, self.S, inverter=self.inverter)

    @property
    @memo
    def value(self):
        energy = 0.5 * self.position.vdot(self._t1)
        energy += 0.5 * self.residual.vdot(self._t2)
        return energy.real

    @property
    @memo
    def gradient(self):
        gradient = self._t1.copy()
        gradient -= self.LinearizedResponse.adjoint_times(self._t2)
        return gradient

    @property
    @memo
    def curvature(self):
        curvature = NonlinearSignalCurvature(R=self.LinearizedResponse,
                                             N=self.N,
                                             S=self.S, inverter=self.inverter)
        return curvature
