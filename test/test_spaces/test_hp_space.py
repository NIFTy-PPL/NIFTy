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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest
from nifty8 import HPSpace, myassert
from numpy.testing import assert_almost_equal, assert_equal, assert_raises

from ..common import setup_function, teardown_function

pmp = pytest.mark.parametrize
# [nside, expected]
CONSTRUCTOR_CONFIGS = [[
    2, {
        'nside': 2,
        'harmonic': False,
        'shape': (48,),
        'size': 48,
    }
], [5, {
    'nside': 5,
    'harmonic': False,
    'shape': (300,),
    'size': 300,
}], [1, {
    'nside': 1,
    'harmonic': False,
    'shape': (12,),
    'size': 12,
}], [0, {
    'error': ValueError
}]]


def test_property_ret_type():
    x = HPSpace(2)
    myassert(isinstance(getattr(x, 'nside'), int))


@pmp('nside, expected', CONSTRUCTOR_CONFIGS)
def test_constructor(nside, expected):
    if 'error' in expected:
        with assert_raises(expected['error']):
            HPSpace(nside)
    else:
        h = HPSpace(nside)
        for key, value in expected.items():
            assert_equal(getattr(h, key), value)


def test_dvol():
    assert_almost_equal(HPSpace(2).dvol, np.pi/12)
