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

import numpy as np
import pytest

import nifty7 as ift

from ..common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize

@pmp("op", [ift.LaplaceOperator, ift.UniformOperator])
@pmp("loc", [0, 1.324, 1400])
@pmp("scale", [1., 0.0929, 1312.19])
def test_inverse(op, loc, scale):
    dom = ift.UnstructuredDomain([10])
    op = op(dom, loc=loc, scale=scale)
    pos = ift.from_random(dom)
    pos1 = op.inverse(op(pos))
    ift.extra.assert_allclose(pos, pos1)
