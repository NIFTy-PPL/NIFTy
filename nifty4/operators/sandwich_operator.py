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

from .endomorphic_operator import EndomorphicOperator
import numpy as np


class SandwichOperator(EndomorphicOperator):
    """Operator which is equivalent to the expression `bun.adjoint*cheese*bun`.

    Parameters
    ----------
    bun: LinearOperator
        the bun part
    cheese: EndomorphicOperator
        the cheese part
    """

    def __init__(self, bun, cheese):
        super(SandwichOperator, self).__init__()
        self._bun = bun
        self._cheese = cheese
        self._op = bun.adjoint*cheese*bun

    @property
    def domain(self):
        return self._op.domain

    @property
    def capability(self):
        return self._op.capability

    def apply(self, x, mode):
        return self._op.apply(x, mode)

    def draw_sample(self, dtype=np.float64):
        return self._bun.adjoint_times(self._cheese.draw_sample(dtype))
