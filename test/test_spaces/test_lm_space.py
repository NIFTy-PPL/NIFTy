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

import unittest
from test.common import expand

import nifty5 as ift
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal, assert_raises

# [lmax, expected]
CONSTRUCTOR_CONFIGS = [
        [5, None, {
            'lmax': 5,
            'mmax': 5,
            'shape': (36,),
            'harmonic': True,
            'size': 36,
            }],
        [7, 4, {
            'lmax': 7,
            'mmax': 4,
            'shape': (52,),
            'harmonic': True,
            'size': 52,
            }],
        [-1, 28, {
            'error': ValueError
            }]
    ]


def _k_length_array_helper(index_arr, lmax):
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


def get_k_length_array_configs():
    da_0 = [_k_length_array_helper(idx, 5) for idx in np.arange(36)]
    return [[5, da_0]]


class LMSpaceInterfaceTests(unittest.TestCase):
    @expand([['lmax', int],
            ['mmax', int],
            ['size', int]])
    def test_property_ret_type(self, attribute, expected_type):
        l = ift.LMSpace(7, 5)
        assert_(isinstance(getattr(l, attribute), expected_type))


class LMSpaceFunctionalityTests(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, lmax, mmax, expected):
        if 'error' in expected:
            with assert_raises(expected['error']):
                ift.LMSpace(lmax, mmax)
        else:
            l = ift.LMSpace(lmax, mmax)
            for key, value in expected.items():
                assert_equal(getattr(l, key), value)

    def test_dvol(self):
        assert_allclose(ift.LMSpace(5).dvol, 1.)

    @expand(get_k_length_array_configs())
    def test_k_length_array(self, lmax, expected):
        l = ift.LMSpace(lmax)
        assert_allclose(l.get_k_length_array().to_global_data(), expected)
