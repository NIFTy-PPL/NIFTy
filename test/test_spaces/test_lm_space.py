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

from numpy.testing import assert_, assert_equal, assert_raises,\
        assert_almost_equal
from nose.plugins.skip import SkipTest
from d2o import distributed_data_object
from nifty import LMSpace
from nifty.config import dependency_injector as di
from test.common import expand

# [lmax, dtype, expected]
CONSTRUCTOR_CONFIGS = [
        [5, None, {
            'lmax': 5,
            'mmax': 5,
            'shape': (36,),
            'harmonic': True,
            'dim': 36,
            'total_volume': 36.0,
            'dtype': np.dtype('float64')
            }],
        [7, np.dtype('float64'), {
            'lmax': 7,
            'mmax': 7,
            'shape': (64,),
            'harmonic': True,
            'dim': 64,
            'total_volume': 64.0,
            'dtype': np.dtype('float64')
            }],
        [-1, None, {
            'error': ValueError
            }]
    ]


def get_distance_array_configs():
    npzfile = np.load('test/data/lm_space.npz')
    return [[5, None, npzfile['da_0']]]


def get_weight_configs():
    npzfile = np.load('test/data/lm_space.npz')
    return [
        [npzfile['w_0_x'], 1, None, False, npzfile['w_0_res']],
        [npzfile['w_0_x'], 1, None, True, npzfile['w_0_res']]
        ]


def get_hermitian_configs():
    npzfile = np.load('test/data/lm_space.npz')
    return [
        [npzfile['h_0_x'], npzfile['h_0_res_real'], npzfile['h_0_res_imag']]
    ]


class LMSpaceIntefaceTests(unittest.TestCase):
    @expand([['lmax', int],
            ['mmax', int],
            ['dim', int]])
    def test_property_ret_type(self, attribute, expected_type):
        try:
            l = LMSpace(7)
        except ImportError:
            raise SkipTest
        assert_(isinstance(getattr(l, attribute), expected_type))


class LMSpaceFunctionalityTests(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, lmax, dtype, expected):
        if 'libsharp_wrapper_gl' not in di or 'healpy' not in di:
            raise SkipTest
        else:
            if 'error' in expected:
                with assert_raises(expected['error']):
                    LMSpace(lmax, dtype)
            else:
                l = LMSpace(lmax, dtype)
                for key, value in expected.iteritems():
                    assert_equal(getattr(l, key), value)

    @expand(get_hermitian_configs())
    def test_hermitian_decomposition(self, x, real, imag):
        if 'libsharp_wrapper_gl' not in di or 'healpy' not in di:
            raise SkipTest
        else:
            l = LMSpace(5)
            assert_almost_equal(
                l.hermitian_decomposition(distributed_data_object(x))[0],
                real)
            assert_almost_equal(
                l.hermitian_decomposition(distributed_data_object(x))[1],
                imag)

    @expand(get_weight_configs())
    def test_weight(self, x, power, axes, inplace, expected):
        if 'libsharp_wrapper_gl' not in di or 'healpy' not in di:
            raise SkipTest
        else:
            l = LMSpace(5)
            res = l.weight(x, power, axes, inplace)
            assert_almost_equal(res, expected)
            if inplace:
                assert_(x is res)

    @expand(get_distance_array_configs())
    def test_distance_array(self, lmax, dtype, expected):
        if 'libsharp_wrapper_gl' not in di or 'healpy' not in di:
            raise SkipTest
        else:
            l = LMSpace(lmax, dtype)
            assert_almost_equal(l.get_distance_array('not').data, expected)
