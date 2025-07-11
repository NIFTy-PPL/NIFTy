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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
import pytest
from nifty8.library.variational_models import LowerTriangularInserter
from numpy.testing import assert_allclose

from ..common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize


@pmp("N", [17, 32])
def test_lower_triangular_inserter(N):
    square_space = ift.RGSpace([N, N])
    op = LowerTriangularInserter(square_space)
    flat_space = op.domain
    assert_allclose(flat_space.shape, (N*(N+1)//2,))

    mat = op(ift.from_random(flat_space))
    for i in range(0,N):
        for j in range(i+1, N):
            assert mat.val[i, j] == 0.
    assert_allclose((mat.val == 0).sum(), (N-1)*N//2)
