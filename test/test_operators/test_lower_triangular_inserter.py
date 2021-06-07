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

from ...nifty7.library.variational_models import LowerTriangularInserter


def test_lower_triangular_inserter():
    N = 42
    square_space = ift.RGSpace([N,N])
    myInserter = LowerTriangularInserter(square_space)
    flat_space = myInserter.domain
    assert_allclose(flat_space.size, N*(N+1)//2)

    myField = ift.from_random(flat_space)
    myMatrix = myInserter(myField)

    assert_allclose(np_selected, selected)

    upper_i = []
    upper_j = []
    for i in range(0,N):
        for j in range(i+1,N):
            upper_i.append(i)
            upper_j.append(j)
    zeros = myMatrix.val[(upper_i,upper_j)]

    assert_allclose(zeros, np.zeros((N-1)*N//2))
    assert_allclose((myMatrix==0).val.sum(), (N-1)*N//2)



    
