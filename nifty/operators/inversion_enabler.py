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
from .linear_operator import LinearOperator


class InversionEnabler(LinearOperator):

    def __init__(self, op, inverter, preconditioner=None):
        self._op = op
        self._inverter = inverter
        if preconditioner is None and hasattr(op, "preconditioner"):
            self._preconditioner = op.preconditioner
        else:
            self._preconditioner = preconditioner
        super(InversionEnabler, self).__init__()

    @property
    def domain(self):
        return self._op.domain

    @property
    def target(self):
        return self._op.target

    @property
    def unitary(self):
        return self._op.unitary

    @property
    def op(self):
        return self._op

    def _times(self, x):
        try:
            res = self._op._times(x)
        except NotImplementedError:
            x0 = Field.zeros(self.target, dtype=x.dtype)
            (result, convergence) = self._inverter(QuadraticEnergy(
                                           A=self._op.inverse_times,
                                           b=x, position=x0),
                                           preconditioner=self._preconditioner)
            res = result.position
        return res

    def _adjoint_times(self, x):
        try:
            res = self._op._adjoint_times(x)
        except NotImplementedError:
            x0 = Field.zeros(self.domain, dtype=x.dtype)
            (result, convergence) = self._inverter(QuadraticEnergy(
                                           A=self.adjoint_inverse_times,
                                           b=x, position=x0),
                                           preconditioner=self._preconditioner)
            res = result.position
        return res

    def _inverse_times(self, x):
        try:
            res = self._op._inverse_times(x)
        except NotImplementedError:
            x0 = Field.zeros(self.domain, dtype=x.dtype)
            (result, convergence) = self._inverter(QuadraticEnergy(
                                           A=self.times,
                                           b=x, position=x0),
                                           preconditioner=self._preconditioner)
            res = result.position
        return res

    def _adjoint_inverse_times(self, x):
        try:
            res = self._op._adjoint_inverse_times(x)
        except NotImplementedError:
            x0 = Field.zeros(self.target, dtype=x.dtype)
            (result, convergence) = self._inverter(QuadraticEnergy(
                                           A=self.adjoint_times,
                                           b=x, position=x0),
                                           preconditioner=self._preconditioner)
            res = result.position
        return res
