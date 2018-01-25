# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from .wiener_filter_curvature import WienerFilterCurvature
from ..utilities import memo
from ..minimization.energy import Energy
from .response_operators import LinearizedSignalResponse


class NonlinearWienerFilterEnergy(Energy):
    def __init__(self, position, d, Instrument, nonlinearity, FFT, power, N, S, sunit,
                 inverter=None):
        super(NonlinearWienerFilterEnergy, self).__init__(position=position)
        self.d = d
        self.sunit = sunit
        self.Instrument = Instrument
        self.nonlinearity = nonlinearity
        self.FFT = FFT
        self.power = power
        position_map = FFT.inverse_times(self.power * self.position)
        self.LinearizedResponse = \
            LinearizedSignalResponse(Instrument, nonlinearity, FFT, power,
                                     position_map, sunit)
        residual = d - Instrument(sunit * nonlinearity(position_map))
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
                              self.S, self.sunit, inverter=self.inverter)

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
