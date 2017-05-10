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

from __future__ import division

import unittest
import numpy as np

from d2o import distributed_data_object
from numpy.testing import assert_, assert_equal, assert_almost_equal,\
        assert_raises
from nifty import PowerSpace, RGSpace, Space
from types import NoneType
from test.common import expand

# [harmonic_domain, distribution_strategy,
#  logarithmic, nbin, binbounds, expected]
CONSTRUCTOR_CONFIGS = [
    [1, 'not', False, None, None, {'error': ValueError}],
    [RGSpace((8,)), 'not', False, None, None, {'error': ValueError}],
    [RGSpace((8,), harmonic=True), 'not', False, None, None, {
        'harmonic': True,
        'shape': (5,),
        'dim': 5,
        'total_volume': 8.0,
        'harmonic_domain': RGSpace((8,), harmonic=True),
        'log': False,
        'nbin': None,
        'binbounds': None,
        'pindex': distributed_data_object([0, 1, 2, 3, 4, 3, 2, 1]),
        'kindex': np.array([0., 1., 2., 3., 4.]),
        'rho': np.array([1, 2, 2, 2, 1]),
        'pundex': np.array([0, 1, 2, 3, 4]),
        'k_array': np.array([0., 1., 2., 3., 4., 3., 2., 1.]),
        }],
    [RGSpace((8,), harmonic=True), 'not', True, None, None, {
        'harmonic': True,
        'shape': (2,),
        'dim': 2,
        'total_volume': 8.0,
        'harmonic_domain': RGSpace((8,), harmonic=True),
        'log': True,
        'nbin': None,
        'binbounds': None,
        'pindex': distributed_data_object([0, 1, 1, 1, 1, 1, 1, 1]),
        'kindex': np.array([0., 2.28571429]),
        'rho': np.array([1, 7]),
        'pundex': np.array([0, 1]),
        'k_array': np.array([0., 2.28571429, 2.28571429, 2.28571429,
                             2.28571429, 2.28571429, 2.28571429, 2.28571429]),
        }],
    ]


def get_distance_array_configs():
    da_0 = np.array([0, 1.0, 1.41421356, 2., 2.23606798, 2.82842712])
    return [
        [RGSpace((4, 4), harmonic=True),  da_0],
        ]


def get_weight_configs():
    np.random.seed(42)

    # power 1
    w_0_x = np.random.rand(32, 16, 6)
    # RGSpace((4, 4), harmonic=True)
    # using rho directly
    weight_0 = np.array([1, 4, 4, 2, 4, 1])
    weight_0 = weight_0.reshape([1, 1, 6])
    w_0_res = w_0_x * weight_0
    return [
        [RGSpace((4, 4), harmonic=True),
            w_0_x, 1, (2,), False, w_0_res],
        [RGSpace((4, 4), harmonic=True),
            w_0_x.copy(), 1, (2,), True, w_0_res],
        ]


class PowerSpaceInterfaceTest(unittest.TestCase):
    @expand([
        ['harmonic_domain', Space],
        ['logarithmic', bool],
        ['nbin', (int, NoneType)],
        ['binbounds', (list, NoneType)],
        ['pindex', distributed_data_object],
        ['kindex', np.ndarray],
        ['rho', np.ndarray],
        ['pundex', np.ndarray],
        ['k_array', distributed_data_object],
        ])
    def test_property_ret_type(self, attribute, expected_type):
        r = RGSpace((4, 4), harmonic=True)
        p = PowerSpace(r)
        assert_(isinstance(getattr(p, attribute), expected_type))


class PowerSpaceFunctionalityTest(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, harmonic_domain, distribution_strategy,
                         logarithmic, nbin, binbounds, expected):
        if 'error' in expected:
            with assert_raises(expected['error']):
                PowerSpace(harmonic_domain=harmonic_domain,
                           distribution_strategy=distribution_strategy,
                           logarithmic=logarithmic, nbin=nbin,
                           binbounds=binbounds)
        else:
            p = PowerSpace(harmonic_domain=harmonic_domain,
                           distribution_strategy=distribution_strategy,
                           logarithmic=logarithmic, nbin=nbin,
                           binbounds=binbounds)
            for key, value in expected.iteritems():
                if isinstance(value, np.ndarray):
                    assert_almost_equal(getattr(p, key), value)
                else:
                    assert_equal(getattr(p, key), value)

    @expand(get_distance_array_configs())
    def test_distance_array(self, harmonic_domain, expected):
        p = PowerSpace(harmonic_domain=harmonic_domain)
        assert_almost_equal(p.get_distance_array('not'), expected)

    @expand(get_weight_configs())
    def test_weight(self, harmonic_domain, x, power, axes,
                    inplace, expected):
        p = PowerSpace(harmonic_domain=harmonic_domain)
        res = p.weight(x, power, axes, inplace)
        assert_almost_equal(res, expected)
        if inplace:
            assert_(x is res)
