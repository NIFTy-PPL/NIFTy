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

import numpy as np

from .diagonal_operator import DiagonalOperator
from .endomorphic_operator import EndomorphicOperator
from .scaling_operator import ScalingOperator


class SandwichOperator(EndomorphicOperator):
    """Operator which is equivalent to the expression `bun.adjoint*cheese*bun`.

    Parameters
    ----------
    bun: LinearOperator
        the bun part
    cheese: EndomorphicOperator
        the cheese part
    """

    def __init__(self, bun, cheese=None):
        super(SandwichOperator, self).__init__()
        self._bun = bun
        if cheese is None:
            self._cheese = ScalingOperator(1., bun.target)
            self._op = bun.adjoint*bun
        else:
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

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        # Drawing samples from diagonal operators is easy (inverse is possible)
        if isinstance(self._op, (ScalingOperator, DiagonalOperator)):
            return self._op.draw_sample(from_inverse, dtype)

        # Inverse samples from general sandwiches is not possible
        if from_inverse:
            raise NotImplementedError(
                "cannot draw from inverse of this operator")

        # Samples from general sandwiches
        return self._bun.adjoint_times(
            self._cheese.draw_sample(from_inverse, dtype))
