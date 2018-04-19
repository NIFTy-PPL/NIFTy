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
from ..nonlinear.nonlinear_operator import *


class NonlinearWienerFilterEnergy(Energy):
    def __init__(self, position, d, Instrument, nonlinearity, ht, power, N, S,
                 inverter=None):
        super(NonlinearWienerFilterEnergy, self).__init__(position=position)
        self.d = d.lock()
        self.Instrument = Instrument
        self.nonlinearity = nonlinearity
        self.ht = ht
        self.power = power

        pos = NLOp_var()
        m = NLOp_Linop(ht, power*pos)
        residual = d - NLOp_Linop(Instrument, NLOp_Tanh(m))
        ene = 0.5 * (NLOp_vdot(pos, NLOp_Linop(S.inverse, pos)) +
                     NLOp_vdot(residual, NLOp_Linop(N.inverse, residual)))
        self._value = ene.value(self.position)
        self._gradient = ene.derivative(self.position)(ift.Field((),1.))

        m = ht(power*position)

        self.N = N
        self.S = S
        self.inverter = inverter
        self.R = Instrument * nonlinearity.derivative(m) * ht * power

    def at(self, position):
        return self.__class__(position, self.d, self.Instrument,
                              self.nonlinearity, self.ht, self.power, self.N,
                              self.S, self.inverter)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @property
    @memo
    def curvature(self):
        return WienerFilterCurvature(self.R, self.N, self.S, self.inverter)
