# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
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

import unittest
import numpy as np
import itertools

from numpy.testing import assert_, assert_equal, assert_raises,\
        assert_almost_equal
from nose.plugins.skip import SkipTest
from nifty import GLSpace
from nifty.config import dependency_injector as di
from test.common import expand

# [nlat, nlon, expected]
CONSTRUCTOR_CONFIGS = [
        [2, None, {
            'nlat': 2,
            'nlon': 3,
            'harmonic': False,
            'shape': (6,),
            'dim': 6,
            'total_volume': 4 * np.pi
            }],
        [0, None, {
            'error': ValueError
            }]
    ]


def get_weight_configs():
    np.random.seed(42)
    wgt = [2.0943951,  2.0943951]
    # for GLSpace(nlat=2, nlon=3)
    weight_0 = np.array(list(itertools.chain.from_iterable(
        itertools.repeat(x, 3) for x in wgt)))
    w_0_x = np.random.rand(6)
    w_0_res = w_0_x * weight_0

    weight_1 = np.array(list(itertools.chain.from_iterable(
        itertools.repeat(x, 3) for x in wgt)))
    weight_1 = weight_1.reshape([1, 1, 6])
    w_1_x = np.random.rand(32, 16, 6)
    w_1_res = w_1_x * weight_1
    return [
        [w_0_x, 1, None, False, w_0_res],
        [w_0_x.copy(), 1, None, True, w_0_res],
        [w_1_x.copy(), 1, (2,), True, w_1_res],
        ]


class GLSpaceInterfaceTests(unittest.TestCase):
    @expand([['nlat', int],
            ['nlon', int]])
    def test_property_ret_type(self, attribute, expected_type):
        try:
            g = GLSpace(2)
        except ImportError:
            raise SkipTest
        else:
            assert_(isinstance(getattr(g, attribute), expected_type))


class GLSpaceFunctionalityTests(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, nlat, nlon, expected):
        try:
            g = GLSpace(4)
        except ImportError:
            raise SkipTest

        if 'error' in expected:
            with assert_raises(expected['error']):
                GLSpace(nlat, nlon)
        else:
            g = GLSpace(nlat, nlon)
            for key, value in expected.iteritems():
                assert_equal(getattr(g, key), value)

    @expand(get_weight_configs())
    def test_weight(self, x, power, axes, inplace, expected):
        try:
            g = GLSpace(4)
        except ImportError:
            raise SkipTest

        if 'pyHealpix' not in di:
            raise SkipTest
        else:
            g = GLSpace(2)
            res = g.weight(x, power, axes, inplace)
            assert_almost_equal(res, expected)
            if inplace:
                assert_(x is res)
