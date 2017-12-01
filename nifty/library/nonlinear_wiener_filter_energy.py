from .wiener_filter_curvature import WienerFilterCurvature
from ..utilities import memo
from ..minimization.energy import Energy
from .response_operators import LinearizedSignalResponse


class NonlinearWienerFilterEnergy(Energy):
    def __init__(self, position, d, Instrument, nonlinearity, FFT, power, N, S,
                 inverter=None):
        super(NonlinearWienerFilterEnergy, self).__init__(position=position)
        self.d = d
        self.Instrument = Instrument
        self.nonlinearity = nonlinearity
        self.FFT = FFT
        self.power = power
        self.LinearizedResponse = \
            LinearizedSignalResponse(Instrument, nonlinearity, FFT, power,
                                     self.position)

        position_map = FFT.adjoint_times(self.power * self.position)
        residual = d - Instrument(nonlinearity(position_map))
        self.N = N
        self.S = S
        self.inverter = inverter
        t1 = self.S.inverse_times(self.position)
        t2 = self.N.inverse_times(residual)
        tmp = self.position.vdot(t1) + residual.vdot(t2)
        self._value = 0.5 * tmp.real
        self._gradient = t1 - self.LinearizedResponse.adjoint_times(t2)

    def at(self, position):
        return self.__class__(position, self.d, self.Instrument,
                              self.nonlinearity, self.FFT, self.power, self.N,
                              self.S, inverter=self.inverter)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @property
    @memo
    def curvature(self):
        return WienerFilterCurvature(R=self.LinearizedResponse, N=self.N,
                                     S=self.S, inverter=self.inverter)
