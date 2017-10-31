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

from numpy.testing import assert_, assert_equal, assert_almost_equal
from nifty2go import RGSpace
from test.common import expand

# [shape, distances, harmonic, expected]
CONSTRUCTOR_CONFIGS = [
        [(8,), None, False,
            {
                'shape': (8,),
                'distances': (0.125,),
                'harmonic': False,
                'dim': 8,
            }],
        [(8,), None, True,
            {
                'shape': (8,),
                'distances': (1.0,),
                'harmonic': True,
                'dim': 8,
            }],
        [(8,), (12,), True,
            {
                'shape': (8,),
                'distances': (12.0,),
                'harmonic': True,
                'dim': 8,
            }],
        [(11, 11), None, False,
            {
                'shape': (11, 11),
                'distances': (1/11, 1/11),
                'harmonic': False,
                'dim': 121,
            }],
        [(12, 12), (1.3, 1.3), True,
            {
                'shape': (12, 12),
                'distances': (1.3, 1.3),
                'harmonic': True,
                'dim': 144,
            }]

    ]


def get_k_length_array_configs():
    # for RGSpace(shape=(4, 4), distances=(0.25,0.25))
    cords_0 = np.ogrid[0:4, 0:4]
    da_0 = ((cords_0[0] - 4 // 2) * 0.25)**2
    da_0 = np.fft.ifftshift(da_0)
    temp = ((cords_0[1] - 4 // 2) * 0.25)**2
    temp = np.fft.ifftshift(temp)
    da_0 = da_0 + temp
    da_0 = np.sqrt(da_0)
    return [
        [(4, 4), (0.25, 0.25), da_0],
        ]


def get_dvol_configs():
    np.random.seed(42)
    return [
        [(11, 11), None, False, 1],
        [(11, 11), None, False, 1],
        [(11, 11), (1.3, 1.3), False, 1],
        [(11, 11), (1.3, 1.3), False, 1],
        [(11, 11), None, True, 1],
        [(11, 11), None, True, 1],
        [(11, 11), (1.3, 1.3), True, 1],
        [(11, 11), (1.3, 1.3), True, 1]
        ]


class RGSpaceInterfaceTests(unittest.TestCase):
    @expand([['distances', tuple]])
    def test_property_ret_type(self, attribute, expected_type):
        x = RGSpace(1)
        assert_(isinstance(getattr(x, attribute), expected_type))


class RGSpaceFunctionalityTests(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, shape, distances,
                         harmonic, expected):
        x = RGSpace(shape, distances, harmonic)
        for key, value in expected.items():
            assert_equal(getattr(x, key), value)

    @expand(get_k_length_array_configs())
    def test_k_length_array(self, shape, distances, expected):
        r = RGSpace(shape=shape, distances=distances, harmonic=True)
        assert_almost_equal(r.get_k_length_array().val, expected)

    @expand(get_dvol_configs())
    def test_dvol(self, shape, distances, harmonic, power):
        r = RGSpace(shape=shape, distances=distances, harmonic=harmonic)
        assert_almost_equal(r.dvol(), np.prod(r.distances)**power)
