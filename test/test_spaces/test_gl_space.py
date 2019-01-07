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

import itertools
import unittest
from test.common import expand

import numpy as np
from nifty5 import GLSpace
from numpy.testing import (assert_, assert_almost_equal, assert_equal,
                           assert_raises)

# [nlat, nlon, expected]
CONSTRUCTOR_CONFIGS = [
        [2, None, {
            'nlat': 2,
            'nlon': 3,
            'harmonic': False,
            'shape': (6,),
            'size': 6,
            }],
        [0, None, {
            'error': ValueError
            }]
    ]


def get_dvol_configs():
    np.random.seed(42)
    wgt = [2.0943951,  2.0943951]
    # for GLSpace(nlat=2, nlon=3)
    dvol_0 = np.array(list(itertools.chain.from_iterable(
        itertools.repeat(x, 3) for x in wgt)))
    return [
        [1, dvol_0],
        ]


class GLSpaceInterfaceTests(unittest.TestCase):
    @expand([['nlat', int],
            ['nlon', int]])
    def test_property_ret_type(self, attribute, expected_type):
        g = GLSpace(2)
        assert_(isinstance(getattr(g, attribute), expected_type))


class GLSpaceFunctionalityTests(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, nlat, nlon, expected):
        g = GLSpace(4)

        if 'error' in expected:
            with assert_raises(expected['error']):
                GLSpace(nlat, nlon)
        else:
            g = GLSpace(nlat, nlon)
            for key, value in expected.items():
                assert_equal(getattr(g, key), value)

    @expand(get_dvol_configs())
    def test_dvol(self, power, expected):
        assert_almost_equal(GLSpace(2).dvol, expected)
