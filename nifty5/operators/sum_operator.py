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
from ..compat import *
from ..utilities import my_sum
from .linear_operator import LinearOperator
import numpy as np


class SumOperator(LinearOperator):
    """Class representing sums of operators."""

    def __init__(self, ops, neg, dom, tgt, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        super(SumOperator, self).__init__()
        self._ops = ops
        self._neg = neg
        self._domain = dom
        self._target = tgt
        self._capability = self.TIMES | self.ADJOINT_TIMES
        for op in ops:
            self._capability &= op.capability

    @staticmethod
    def simplify(ops, neg):
        from .scaling_operator import ScalingOperator
        from .diagonal_operator import DiagonalOperator
        # Step 1: verify domains
        dom = ops[0].domain
        tgt = ops[0].target
        for op in ops[1:]:
            dom = dom.unitedWith(op.domain)
            tgt = tgt.unitedWith(op.target)

        # Step 2: unpack SumOperators
        opsnew = []
        negnew = []
        for op, ng in zip(ops, neg):
            if isinstance(op, SumOperator):
                opsnew += op._ops
                if ng:
                    negnew += [not n for n in op._neg]
                else:
                    negnew += list(op._neg)
            else:
                opsnew.append(op)
                negnew.append(ng)
        ops = opsnew
        neg = negnew
        # Step 3: collect ScalingOperators
        sum = 0.
        opsnew = []
        negnew = []
        lastdom = ops[-1].domain
        for op, ng in zip(ops, neg):
            if isinstance(op, ScalingOperator):
                sum += op._factor * (-1 if ng else 1)
            else:
                opsnew.append(op)
                negnew.append(ng)
        if sum != 0.:
            # try to absorb the factor into a DiagonalOperator
            for i in range(len(opsnew)):
                if isinstance(opsnew[i], DiagonalOperator):
                    sum *= (-1 if negnew[i] else 1)
                    opsnew[i] = opsnew[i]._add(sum)
                    sum = 0.
                    break
        if sum != 0:
            # have to add the scaling operator at the end
            opsnew.append(ScalingOperator(sum, lastdom))
            negnew.append(False)
        ops = opsnew
        neg = negnew
        # Step 4: combine DiagonalOperators where possible
        processed = [False] * len(ops)
        opsnew = []
        negnew = []
        for i in range(len(ops)):
            if not processed[i]:
                if isinstance(ops[i], DiagonalOperator):
                    op = ops[i]
                    opneg = neg[i]
                    for j in range(i+1, len(ops)):
                        if isinstance(ops[j], DiagonalOperator):
                            op = op._combine_sum(ops[j], opneg, neg[j])
                            opneg = False
                            processed[j] = True
                    opsnew.append(op)
                    negnew.append(opneg)
                else:
                    opsnew.append(ops[i])
                    negnew.append(neg[i])
        ops = opsnew
        neg = negnew
        # Step 5: combine BlockDiagonalOperators where possible
        from ..multi.block_diagonal_operator import BlockDiagonalOperator
        processed = [False] * len(ops)
        opsnew = []
        negnew = []
        for i in range(len(ops)):
            if not processed[i]:
                if isinstance(ops[i], BlockDiagonalOperator):
                    op = ops[i]
                    opneg = neg[i]
                    for j in range(i+1, len(ops)):
                        if isinstance(ops[j], BlockDiagonalOperator):
                            op = op._combine_sum(ops[j], opneg, neg[j])
                            opneg = False
                            processed[j] = True
                    opsnew.append(op)
                    negnew.append(opneg)
                else:
                    opsnew.append(ops[i])
                    negnew.append(neg[i])
        ops = opsnew
        neg = negnew
        return ops, neg, dom, tgt

    @staticmethod
    def make(ops, neg):
        """Build a SumOperator (or something simpler if possible)

        Parameters
        ----------
        ops: list of LinearOperator
            Individual operators of the sum.
        neg: list of bool
            Same length as ops.
            If True then the equivalent operator gets a minus in the sum.
        """
        ops = tuple(ops)
        neg = tuple(neg)
        if len(ops) == 0:
            raise ValueError("ops is empty")
        if len(ops) != len(neg):
            raise ValueError("length mismatch between ops and neg")
        ops, neg, dom, tgt = SumOperator.simplify(ops, neg)
        if len(ops) == 1 and not neg[0]:
            return ops[0]
        return SumOperator(ops, neg, dom, tgt, _callingfrommake=True)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def adjoint(self):
        return self.make([op.adjoint for op in self._ops], self._neg)

    @property
    def capability(self):
        return self._capability

    def apply(self, x, mode):
        self._check_mode(mode)
        res = None
        for op, neg in zip(self._ops, self._neg):
            if res is None:
                res = -op.apply(x, mode) if neg else op.apply(x, mode)
            else:
                if neg:
                    res = res - op.apply(x, mode)
                else:
                    res = res + op.apply(x, mode)
        return res

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        if from_inverse:
            raise NotImplementedError(
                "cannot draw from inverse of this operator")
        return my_sum(map(lambda op: op.draw_sample(from_inverse, dtype),
                          self._ops))
