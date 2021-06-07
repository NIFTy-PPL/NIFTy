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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
from numpy.testing import assert_allclose

import nifty7 as ift

from nifty7.library.variational_models import DiagonalSelector


def test_diagonal_selector():
    N = 42
    square_space = ift.RGSpace([N,N])
    linear_space = ift.RGSpace(N)
    myField = ift.from_random(square_space)
    myDiagonalSelector = DiagonalSelector(square_space)
    selected = myDiagonalSelector(myField).val
    np_selected = np.diag(myField.val)
    assert_allclose(np_selected, selected)
    
