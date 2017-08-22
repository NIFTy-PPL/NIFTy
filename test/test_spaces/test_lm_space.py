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
        assert_almost_equal, assert_array_almost_equal
from d2o import distributed_data_object
from nifty import LMSpace
from test.common import expand

# [lmax, expected]
CONSTRUCTOR_CONFIGS = [
        [5, {
            'lmax': 5,
            'mmax': 5,
            'shape': (36,),
            'harmonic': True,
            'dim': 36,
            'total_volume': 36.0,
            }],
        [7, {
            'lmax': 7,
            'mmax': 7,
            'shape': (64,),
            'harmonic': True,
            'dim': 64,
            'total_volume': 64.0,
            }],
        [-1, {
            'error': ValueError
            }]
    ]


def _distance_array_helper(index_arr, lmax):
    if index_arr <= lmax:
        index_half = index_arr
    else:
        if (index_arr - lmax) % 2 == 0:
            index_half = (index_arr + lmax)//2
        else:
            index_half = (index_arr + lmax + 1)//2

    m = np.ceil(((2*lmax + 1) - np.sqrt((2*lmax + 1)**2 -
                 8*(index_half - lmax)))/2).astype(int)

    return index_half - m*(2*lmax + 1 - m)//2


def get_distance_array_configs():
    da_0 = [_distance_array_helper(idx, 5) for idx in np.arange(36)]
    return [[5, da_0]]


def get_weight_configs():
    np.random.seed(42)
    w_0_x = np.random.rand(32, 16, 6)
    return [
        [w_0_x, 1, None, False, w_0_x],
        [w_0_x.copy(), 1, None,  True, w_0_x]
        ]


def get_hermitian_configs():
    np.random.seed(42)
    h_0_res_real = np.random.rand(32, 16, 6).astype(np.complex128)
    h_0_res_imag = np.random.rand(32, 16, 6).astype(np.complex128)
    h_0_x = h_0_res_real + h_0_res_imag * 1j
    return [
        [h_0_x, h_0_res_real, h_0_res_imag]
    ]


class LMSpaceInterfaceTests(unittest.TestCase):
    @expand([['lmax', int],
            ['mmax', int],
            ['dim', int]])
    def test_property_ret_type(self, attribute, expected_type):
        l = LMSpace(7)
        assert_(isinstance(getattr(l, attribute), expected_type))


class LMSpaceFunctionalityTests(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, lmax, expected):
        if 'error' in expected:
            with assert_raises(expected['error']):
                LMSpace(lmax)
        else:
            l = LMSpace(lmax)
            for key, value in expected.items():
                assert_equal(getattr(l, key), value)

    def test_hermitianize_inverter(self):
        l = LMSpace(5)
        v = distributed_data_object(global_shape=l.shape, dtype=np.complex128)
        v[:] = np.random.random(l.shape) + 1j*np.random.random(l.shape)
        inverted = l.hermitianize_inverter(v, axes=(0,))
        assert_array_almost_equal(inverted.get_full_data(), v.get_full_data())

    @expand(get_weight_configs())
    def test_weight(self, x, power, axes, inplace, expected):
        l = LMSpace(5)
        res = l.weight(x, power, axes, inplace)
        assert_almost_equal(res, expected)
        if inplace:
            assert_(x is res)

    @expand(get_distance_array_configs())
    def test_distance_array(self, lmax, expected):
        l = LMSpace(lmax)
        assert_almost_equal(l.get_distance_array('not').data, expected)
