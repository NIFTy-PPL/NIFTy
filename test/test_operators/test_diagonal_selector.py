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
from nifty8.library.variational_models import DiagonalSelector
from numpy.testing import assert_allclose

from ..common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize


@pmp("N", [17, 32])
def test_diagonal_selector(N):
    square_space = ift.RGSpace([N, N])
    linear_space = ift.RGSpace(N)
    myField = ift.from_random(square_space)
    myDiagonalSelector = DiagonalSelector(square_space)
    selected = myDiagonalSelector(myField).val
    np_selected = np.diag(myField.val)
    assert_allclose(np_selected, selected)


@pmp("n", [1, 3])
def test_error(n):
    square_space = ift.RGSpace(n*[13])
    with pytest.raises(AssertionError):
        myDiagonalSelector = DiagonalSelector(square_space)
