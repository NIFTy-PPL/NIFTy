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

from __future__ import absolute_import, division, print_function

import numpy as np

from ..compat import *
from ..utilities import my_sum
from .linear_operator import LinearOperator
from ..sugar import domain_union
from ..multi.multi_domain import MultiDomain
from ..multi.multi_field import MultiField


class RelaxedSumOperator(LinearOperator):
    """Class representing sums of operators with compatible domains."""

    def __init__(self, ops):
        super(RelaxedSumOperator, self).__init__()
        self._ops = ops
        self._domain = domain_union([op.domain for op in ops])
        self._target = domain_union([op.target for op in ops])
        self._capability = self.TIMES | self.ADJOINT_TIMES
        for op in ops:
            self._capability &= op.capability

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def adjoint(self):
        return RelaxedSumOperator([op.adjoint for op in self._ops])

    @property
    def capability(self):
        return self._capability

    def apply(self, x, mode):
        self._check_input(x, mode)
        res = None
        for op in self._ops:
            tmp = op.apply(x.extract(op._dom(mode)), mode)
            res = tmp if res is None else res.unite(tmp)
        return res

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        if from_inverse:
            raise NotImplementedError(
                "cannot draw from inverse of this operator")
        res = None
        for op in self._ops:
            tmp = op.draw_sample(from_inverse, dtype)
            res = tmp if res is None else res.unite(tmp)
        return res
