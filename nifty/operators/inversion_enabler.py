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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from ..minimization.quadratic_energy import QuadraticEnergy
from ..field import Field


class InversionEnabler(object):

    def __init__(self, inverter, preconditioner=None):
        super(InversionEnabler, self).__init__()
        self._inverter = inverter
        self._preconditioner = preconditioner

    def _operation(self, x, op, tdom):
        x0 = Field.zeros(tdom, dtype=x.dtype)
        energy = QuadraticEnergy(A=op, b=x, position=x0)
        r = self._inverter(energy, preconditioner=self._preconditioner)[0]
        return r.position

    def _times(self, x):
        return self._operation(x, self._inverse_times, self.target)

    def _adjoint_times(self, x):
        return self._operation(x, self._adjoint_inverse_times, self.domain)

    def _inverse_times(self, x):
        return self._operation(x, self._times, self.domain)

    def _adjoint_inverse_times(self, x):
        return self._operation(x, self._adjoint_times, self.target)
