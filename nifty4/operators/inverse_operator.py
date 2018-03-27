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

from .linear_operator import LinearOperator
import numpy as np


class InverseOperator(LinearOperator):
    """Adapter class representing the inverse of a given operator."""

    def __init__(self, op):
        super(InverseOperator, self).__init__()
        self._op = op

    @property
    def domain(self):
        return self._op.target

    @property
    def target(self):
        return self._op.domain

    @property
    def capability(self):
        return self._inverseCapability[self._op.capability]

    @property
    def inverse(self):
        return self._op

    def apply(self, x, mode):
        return self._op.apply(x, self._inverseMode[mode])

    def draw_sample(self, dtype=np.float64):
        return self._op.inverse_draw_sample(dtype)

    def inverse_draw_sample(self, dtype=np.float64):
        return self._op.draw_sample(dtype)
