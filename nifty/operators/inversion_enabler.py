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

from ..minimization.quadratic_energy import QuadraticEnergy
from ..minimization.iteration_controller import IterationController
from ..field import Field, dobj
from .linear_operator import LinearOperator


class InversionEnabler(LinearOperator):
    def __init__(self, op, inverter, preconditioner=None):
        super(InversionEnabler, self).__init__()
        self._op = op
        self._inverter = inverter
        self._preconditioner = preconditioner

    @property
    def domain(self):
        return self._op.domain

    @property
    def target(self):
        return self._op.target

    @property
    def capability(self):
        return self._addInverse[self._op.capability]

    def apply(self, x, mode):
        self._check_mode(mode)
        if self._op.capability & mode:
            return self._op.apply(x, mode)

        tdom = self._tgt(mode)
        x0 = Field.zeros(tdom, dtype=x.dtype)

        def func(x):
            return self._op.apply(x, self._inverseMode[mode])

        energy = QuadraticEnergy(A=func, b=x, position=x0)
        r, stat = self._inverter(energy, preconditioner=self._preconditioner)
        if stat != IterationController.CONVERGED:
            dobj.mprint("Error detected during operator inversion")
        return r.position
