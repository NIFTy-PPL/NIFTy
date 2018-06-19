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

from ..minimization import Energy
from ..operators import InversionEnabler, SandwichOperator
from ..utilities import memo


class UnitLogGauss(Energy):
    def __init__(self, s, inverter=None):
        """
        s: Sky model object

        value = 0.5 * s.vdot(s), i.e. a log-Gauss distribution with unit
        covariance
        """
        super(UnitLogGauss, self).__init__(s.position)
        self._s = s
        self._inverter = inverter

    def at(self, position):
        return self.__class__(self._s.at(position), self._inverter)

    @property
    @memo
    def _gradient_helper(self):
        return self._s.gradient

    @property
    @memo
    def value(self):
        return .5 * self._s.value.vdot(self._s.value)

    @property
    @memo
    def gradient(self):
        return self._gradient_helper.adjoint(self._s.value)

    @property
    @memo
    def curvature(self):
        c = SandwichOperator.make(self._gradient_helper)
        if self._inverter is None:
            return c
        return InversionEnabler(c, self._inverter)
