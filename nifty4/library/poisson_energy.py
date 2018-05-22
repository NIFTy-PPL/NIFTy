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

from ..minimization.energy import Energy
from ..operators.diagonal_operator import DiagonalOperator
from ..operators.sandwich_operator import SandwichOperator
from ..operators.inversion_enabler import InversionEnabler
from ..field import log


class PoissonEnergy(Energy):
    def __init__(self, position, d, Instrument, nonlinearity, ht, Phi_h,
                 inverter=None):
        super(PoissonEnergy, self).__init__(position=position)
        self._inverter = inverter
        self._d = d
        self._Instrument = Instrument
        self._nonlinearity = nonlinearity
        self._ht = ht
        self._Phi_h = Phi_h
        htpos = ht(position)
        lam = Instrument(nonlinearity(htpos))
        Rho = DiagonalOperator(nonlinearity.derivative(htpos))
        eps = 1e-100  # to catch harmless 0/0 where data is zero
        W = DiagonalOperator((d+eps)/(lam**2+eps))

        phipos = Phi_h.inverse_adjoint_times(position)
        prior_energy = 0.5*position.vdot(phipos)
        likel_energy = lam.sum()-d.vdot(log(lam+eps))
        self._value = prior_energy + likel_energy

        R1 = Instrument*Rho*ht
        self._grad = (phipos + R1.adjoint_times((lam-d)/(lam+eps))).lock()
        self._curv = Phi_h.inverse + SandwichOperator.make(R1, W)

    def at(self, position):
        return self.__class__(position, self._d, self._Instrument,
                              self._nonlinearity, self._ht, self._Phi_h,
                              self._inverter)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._grad

    @property
    def curvature(self):
        return InversionEnabler(self._curv, self._inverter,
                                approximation=self._Phi_h.inverse)
