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
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
import pytest

from ..common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize


@pmp("op", [ift.LaplaceOperator, ift.UniformOperator])
@pmp("loc", [0, 1.324, 1400])
@pmp("scale", [1.0, 0.0929, 1312.19])
def test_inverse(op, loc, scale):
    dom = ift.UnstructuredDomain([10])
    op = op(dom, loc=loc, scale=scale)
    pos = ift.from_random(dom)
    pos1 = op.inverse(op(pos))
    ift.extra.assert_allclose(pos, pos1)


@pmp("alpha", [1.1, 2, 3, 4])
@pmp("q", [0.2, 1, 5, 10])
@pmp("mode", [0.1, 1])
@pmp("mean", [1.25, 3])
def test_init_parameter_equality(alpha, q, mode, mean):
    dom = ift.UnstructuredDomain([10])
    op1 = ift.InverseGammaOperator(dom, alpha=alpha, q=q)
    op2 = ift.InverseGammaOperator(dom, mode=op1.mode, mean=op1.mean)
    op3 = ift.InverseGammaOperator(dom, mode=mode, mean=mean)
    op4 = ift.InverseGammaOperator(dom, alpha=op3.alpha, q=op3.q)
    f = ift.from_random(dom)
    pos1 = op1(f)
    pos2 = op2(f)
    pos3 = op3(f)
    pos4 = op4(f)
    ift.extra.assert_allclose(pos1, pos2)
    ift.extra.assert_allclose(pos3, pos4)
