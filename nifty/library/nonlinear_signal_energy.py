from .wiener_filter_curvature import WienerFilterCurvature
from .. import Field, exp
from ..utilities import memo
from ..sugar import generate_posterior_sample
from ..minimization.energy import Energy
from ..operators.inversion_enabler import InversionEnabler
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
        return self._t1 - self.LinearizedResponse.adjoint_times(self._t2)

    @property
    @memo
    def curvature(self):
        curvature = WienerFilterCurvature(R=self.LinearizedResponse,
                                          N=self.N, S=self.S)
        return InversionEnabler(curvature, inverter=self.inverter)
