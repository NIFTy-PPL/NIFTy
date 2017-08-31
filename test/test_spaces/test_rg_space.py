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

from numpy.testing import assert_, assert_equal, assert_almost_equal, \
                          assert_array_equal
from nifty import RGSpace
from test.common import expand
from itertools import product
from nose.plugins.skip import SkipTest

# [shape, distances, harmonic, expected]
CONSTRUCTOR_CONFIGS = [
        [(8,), None, False,
            {
                'shape': (8,),
                'distances': (0.125,),
                'harmonic': False,
                'dim': 8,
                'total_volume': 1.0
            }],
        [(8,), None, True,
            {
                'shape': (8,),
                'distances': (1.0,),
                'harmonic': True,
                'dim': 8,
                'total_volume': 8.0
            }],
        [(8,), (12,), True,
            {
                'shape': (8,),
                'distances': (12.0,),
                'harmonic': True,
                'dim': 8,
                'total_volume': 96.0
            }],
        [(11, 11), None, False,
            {
                'shape': (11, 11),
                'distances': (1/11, 1/11),
                'harmonic': False,
                'dim': 121,
                'total_volume': 1.0
            }],
        [(12, 12), (1.3, 1.3), True,
            {
                'shape': (12, 12),
                'distances': (1.3, 1.3),
                'harmonic': True,
                'dim': 144,
                'total_volume': 243.36
            }]

    ]


def get_distance_array_configs():
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


def get_weight_configs():
    np.random.seed(42)
    # power 1
    w_0_x = np.random.rand(32, 12, 6)
    # for RGSpace(shape=(11,11), distances=None, harmonic=False)
    w_0_res = w_0_x * (1/11 * 1/11)
    # for RGSpace(shape=(11, 11), distances=(1.3,1.3), harmonic=False)
    w_1_res = w_0_x * (1.3 * 1.3)
    # for RGSpace(shape=(11,11), distances=None, harmonic=True)
    w_2_res = w_0_x * (1.0 * 1.0)
    # for RGSpace(shape=(11,11), distances=(1.3, 1,3), harmonic=True)
    w_3_res = w_0_x * (1.3 * 1.3)
    return [
        [(11, 11), None, False, w_0_x, 1, None, False, w_0_res],
        [(11, 11), None, False, w_0_x.copy(), 1, None,  True, w_0_res],
        [(11, 11), (1.3, 1.3), False, w_0_x, 1, None, False, w_1_res],
        [(11, 11), (1.3, 1.3), False, w_0_x.copy(), 1, None,  True, w_1_res],
        [(11, 11), None, True, w_0_x, 1, None, False, w_2_res],
        [(11, 11), None, True, w_0_x.copy(), 1, None,  True, w_2_res],
        [(11, 11), (1.3, 1.3), True, w_0_x, 1, None, False, w_3_res],
        [(11, 11), (1.3, 1.3), True, w_0_x.copy(), 1, None,  True, w_3_res]
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

    @expand(product([(10,), (11,), (1, 1), (4, 4), (5, 7), (8, 12), (7, 16),
                     (4, 6, 8), (17, 5, 3)]))
    def test_hermitianize_inverter(self, shape):
        try:
            r = RGSpace(shape, harmonic=True)
        except ValueError:
            raise SkipTest
        v = np.empty(shape, dtype=np.complex128)
        v[:] = np.random.random(shape) + 1j*np.random.random(shape)

        inverted = r.hermitianize_inverter(v, axes=range(len(shape)))

        assert_array_equal(v, inverted)

    @expand(get_distance_array_configs())
    def test_distance_array(self, shape, distances, expected):
        r = RGSpace(shape=shape, distances=distances, harmonic=True)
        assert_almost_equal(r.get_distance_array(), expected)

    @expand(get_weight_configs())
    def test_weight(self, shape, distances, harmonic, x, power, axes,
                    inplace, expected):
        r = RGSpace(shape=shape, distances=distances, harmonic=harmonic)
        res = r.weight(x, power, axes, inplace)
        assert_almost_equal(res, expected)
        if inplace:
            assert_(x is res)
