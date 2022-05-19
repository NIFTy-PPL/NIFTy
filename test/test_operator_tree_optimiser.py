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
# Copyright(C) 2020 Max-Planck-Society
# Author: Rouven Lemmerz
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from copy import deepcopy

import nifty8 as ift
import numpy as np
from numpy.testing import assert_allclose


class CountingOp(ift.Operator):
    # FIXME: Not a LinearOperator since ChainOps not supported yet
    def __init__(self, domain):
        self._domain = self._target = ift.sugar.makeDomain(domain)
        self._count = 0

    def apply(self, x):
        self._count += 1
        return x

    @property
    def count(self):
        return self._count


def test_operator_tree_optimiser():
    dom = ift.RGSpace(10, harmonic=True)
    cop1 = CountingOp(dom)
    op1 = (ift.UniformOperator(dom, -1, 2)@cop1).ducktape('a')
    cop2 = CountingOp(dom)
    op2 = ift.FieldZeroPadder(dom, (11,))@cop2
    cop3 = CountingOp(op2.target)
    op3 = ift.ScalingOperator(op2.target, 3)@cop3
    cop4 = CountingOp(op2.target)
    op4 = ift.ScalingOperator(op2.target, 1.5) @ cop4
    op1 = op1 * op1
    # test layering in between two levels
    op = op3@op2@op1 + op2@op1 + op3@op2@op1 + op2@op1
    op = op + op
    op = op4@(op4@op + op4@op)
    fld = ift.from_random(op.domain, 'normal', np.float64)
    op_orig = deepcopy(op)
    op = ift.operator_tree_optimiser._optimise_operator(op)
    assert_allclose(op(fld).val, op_orig(fld).val, rtol=np.finfo(np.float64).eps)
    ift.myassert(1 == ((cop4.count-1) * cop3.count * cop2.count * cop1.count))
    # test testing
    ift.optimise_operator(op_orig)
