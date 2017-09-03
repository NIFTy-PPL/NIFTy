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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import division
import unittest
import numpy as np

from numpy.testing import assert_, assert_equal, assert_raises,\
                          assert_almost_equal
from nifty2go import HPSpace
from test.common import expand

# [nside, expected]
CONSTRUCTOR_CONFIGS = [
        [2, {
            'nside': 2,
            'harmonic': False,
            'shape': (48,),
            'dim': 48,
            'total_volume': 4 * np.pi,
            }],
        [5, {
            'nside': 5,
            'harmonic': False,
            'shape': (300,),
            'dim': 300,
            'total_volume': 4 * np.pi,
            }],
        [1, {
            'nside': 1,
            'harmonic': False,
            'shape': (12,),
            'dim': 12,
            'total_volume': 4 * np.pi,
            }],
        [0, {
            'error': ValueError
            }]
    ]


def get_weight_configs():
    np.random.seed(42)

    # for HPSpace(nside=2)
    w_0_x = np.random.rand(48)
    w_0_res = w_0_x * ((4 * np.pi) / 48)
    w_1_res = w_0_x * (((4 * np.pi) / 48)**2)
    return [
        [w_0_x, 1, None, False, w_0_res],
        [w_0_x.copy(), 1, None, True, w_0_res],
        [w_0_x, 2, None, False, w_1_res],
        ]


class HPSpaceInterfaceTests(unittest.TestCase):
    @expand([['nside', int]])
    def test_property_ret_type(self, attribute, expected_type):
        x = HPSpace(2)
        assert_(isinstance(getattr(x, attribute), expected_type))


class HPSpaceFunctionalityTests(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, nside, expected):
        if 'error' in expected:
            with assert_raises(expected['error']):
                HPSpace(nside)
        else:
            h = HPSpace(nside)
            for key, value in expected.items():
                assert_equal(getattr(h, key), value)

    @expand(get_weight_configs())
    def test_weight(self, x, power, axes, inplace, expected):
        h = HPSpace(2)
        res = h.weight(x, power, axes, inplace)
        assert_almost_equal(res, expected)
        if inplace:
            assert_(x is res)
