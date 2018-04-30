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

from .multi_linear_operator import MultiLinearOperator
import numpy as np


class MultiSumOperator(MultiLinearOperator):
    """Class representing sums of multi-operators."""

    def __init__(self, ops, neg, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        super(MultiSumOperator, self).__init__()
        self._ops = ops
        self._neg = neg
        self._capability = self.TIMES | self.ADJOINT_TIMES
        for op in ops:
            self._capability &= op.capability

    @staticmethod
    def make(ops, neg):
        ops = tuple(ops)
        neg = tuple(neg)
        if len(ops) != len(neg):
            raise ValueError("length mismatch between ops and neg")
        #ops, neg = MultiSumOperator.simplify(ops, neg)
        if len(ops) == 1 and not neg[0]:
            return ops[0]
        return MultiSumOperator(ops, neg, _callingfrommake=True)

    @property
    def domain(self):
        return self._ops[0].domain

    @property
    def target(self):
        return self._ops[0].target

    @property
    def adjoint(self):
        return self.make([op.adjoint for op in self._ops], self._neg)

    @property
    def capability(self):
        return self._capability

    def apply(self, x, mode):
        self._check_mode(mode)
        for i, op in enumerate(self._ops):
            if i == 0:
                res = -op.apply(x, mode) if self._neg[i] else op.apply(x, mode)
            else:
                if self._neg[i]:
                    res -= op.apply(x, mode)
                else:
                    res += op.apply(x, mode)
        return res

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        if from_inverse:
            raise ValueError("cannot draw from inverse of this operator")
        res = self._ops[0].draw_sample(from_inverse, dtype)
        for op in self._ops[1:]:
            res += op.draw_sample(from_inverse, dtype)
        return res
