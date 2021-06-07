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


def test_multifield2vector():
    myFirstSpace = ift.RGSpace([42,43])
    mySecondSpace = ift.RGSpace([1,2,3])
    myDict = {
        'first': ift.from_random(myFirstSpace),
        'second': ift.from_random(mySecondSpace)
    }
    myMultiField = ift.MultiField.from_dict(myDict)
    myMultifield2Vector = ift.Multifield2Vector(myMultiField.domain)
    myVector =  myMultifield2Vector(myMultiField)
    assert_allclose(myVector.size, myMultiField.size)
    
    myNumpyVector = np.empty(myMultiField.size)
    myNumpyVector[:myFirstSpace.size] = myDict['first'].val.flatten()
    myNumpyVector[myFirstSpace.size:] = myDict['second'].val.flatten()
    assert_allclose(myNumpyVector, myVector.val)

    mySecondMultiField = myMultifield2Vector.adjoint(myVector)

    assert_allclose(mySecondMultiField['first'].val, myMultiField['first'].val)
    assert_allclose(mySecondMultiField['second'].val, myMultiField['second'].val)

    
