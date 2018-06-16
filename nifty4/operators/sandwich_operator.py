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

from .linear_operator import LinearOperator
from .diagonal_operator import DiagonalOperator
from .endomorphic_operator import EndomorphicOperator
from .scaling_operator import ScalingOperator


class SandwichOperator(EndomorphicOperator):
    """Operator which is equivalent to the expression `bun.adjoint*cheese*bun`.
    """

    def __init__(self, bun, cheese, op, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        super(SandwichOperator, self).__init__()
        self._bun = bun
        self._cheese = cheese
        self._op = op

    @staticmethod
    def make(bun, cheese=None):
        """Build a SandwichOperator (or something simpler if possible)

        Parameters
        ----------
        bun: LinearOperator
            the bun part
        cheese: EndomorphicOperator
            the cheese part
        """
        if not isinstance(bun, LinearOperator):
            raise TypeError("bun must be a linear operator")
        if cheese is not None and not isinstance(cheese, LinearOperator):
            raise TypeError("cheese must be a linear operator")
        if cheese is None:
            cheese = ScalingOperator(1., bun.target)
            op = bun.adjoint*bun
        else:
            op = bun.adjoint*cheese*bun

        # if our sandwich is diagonal, we can return immediately
        if isinstance(op, (ScalingOperator, DiagonalOperator)):
            return op
        return SandwichOperator(bun, cheese, op, _callingfrommake=True)

    @property
    def domain(self):
        return self._op.domain

    @property
    def capability(self):
        return self._op.capability

    def apply(self, x, mode):
        return self._op.apply(x, mode)

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        # Inverse samples from general sandwiches is not possible
        if from_inverse:
            if self._bun.capabilities & self._bun.INVERSE_TIMES:
                try:
                    s = self._cheese.draw_sample(from_inverse, dtype)
                    return self._bun.inverse_times(s)
                except NotImplementedError:
                    pass
            raise NotImplementedError(
                "cannot draw from inverse of this operator")

        # Samples from general sandwiches
        return self._bun.adjoint_times(
            self._cheese.draw_sample(from_inverse, dtype))
