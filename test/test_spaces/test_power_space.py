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

from d2o import distributed_data_object
from numpy.testing import assert_, assert_equal, assert_almost_equal,\
        assert_raises
from nifty import PowerSpace, RGSpace, Space, LMSpace
from types import NoneType
from test.common import expand
from itertools import product, chain
#needed to check wether fftw is available
from d2o.config import dependency_injector as gdi

HARMONIC_SPACES = [RGSpace((8,), harmonic=True),
    RGSpace((7,), harmonic=True,zerocenter=True),
    RGSpace((8,), harmonic=True,zerocenter=True),
    RGSpace((7,8), harmonic=True),
    RGSpace((7,8), harmonic=True, zerocenter=True),
    RGSpace((6,6), harmonic=True, zerocenter=True),
    RGSpace((7,5), harmonic=True, zerocenter=True),
    RGSpace((5,5), harmonic=True),
    RGSpace((4,5,7), harmonic=True),
    RGSpace((4,5,7), harmonic=True, zerocenter=True),
    LMSpace(6),
    LMSpace(9)]


#Try all sensible kinds of combinations of spaces, distributuion strategy and
#binning parameters
_maybe_fftw = ["fftw"] if ('pyfftw' in gdi) else []

CONSISTENCY_CONFIGS_IMPLICIT = product(HARMONIC_SPACES, ["not", "equal"] + _maybe_fftw, [None], [None, 3,4], [True, False])
CONSISTENCY_CONFIGS_EXPLICIT = product(HARMONIC_SPACES, ["not", "equal"] + _maybe_fftw, [[0.,1.3]],[None],[False])
CONSISTENCY_CONFIGS = chain(CONSISTENCY_CONFIGS_IMPLICIT, CONSISTENCY_CONFIGS_EXPLICIT)

# [harmonic_partner, distribution_strategy,
#  logarithmic, nbin, binbounds, expected]
CONSTRUCTOR_CONFIGS = [
    [1, 'not', False, None, None, {'error': ValueError}],
    [RGSpace((8,)), 'not', False, None, None, {'error': ValueError}],
    [RGSpace((8,), harmonic=True), 'not', False, None, None, {
        'harmonic': True,
        'shape': (5,),
        'dim': 5,
        'total_volume': 8.0,
        'harmonic_partner': RGSpace((8,), harmonic=True),
        'binbounds': None,
        'pindex': distributed_data_object([0, 1, 2, 3, 4, 3, 2, 1]),
        'kindex': np.array([0., 1., 2., 3., 4.]),
        'rho': np.array([1, 2, 2, 2, 1]),
        }],
    [RGSpace((8,), harmonic=True), 'not', True, None, None, {
        'harmonic': True,
        'shape': (2,),
        'dim': 2,
        'total_volume': 8.0,
        'harmonic_partner': RGSpace((8,), harmonic=True),
        'binbounds': None,
        'pindex': distributed_data_object([0, 1, 1, 1, 1, 1, 1, 1]),
        'kindex': np.array([0., 2.28571429]),
        'rho': np.array([1, 7]),
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
        ['harmonic_partner', Space],
        ['binbounds', NoneType],
        ['pindex', distributed_data_object],
        ['kindex', np.ndarray],
        ['rho', np.ndarray],
        ])
    def test_property_ret_type(self, attribute, expected_type):
        r = RGSpace((4, 4), harmonic=True)
        p = PowerSpace(r)
        assert_(isinstance(getattr(p, attribute), expected_type))

class PowerSpaceConsistencyCheck(unittest.TestCase):
    @expand(CONSISTENCY_CONFIGS)
    def test_rhopindexConsistency(self, harmonic_partner, distribution_strategy,
                         binbounds, nbin,logarithmic):
        p = PowerSpace(harmonic_partner=harmonic_partner,
                       distribution_strategy=distribution_strategy,
                       logarithmic=logarithmic, nbin=nbin,
                       binbounds=binbounds)
        assert_equal(p.pindex.flatten().bincount(), p.rho,
            err_msg='rho is not equal to pindex degeneracy')

class PowerSpaceFunctionalityTest(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, harmonic_partner, distribution_strategy,
                         logarithmic, nbin, binbounds, expected):
        if 'error' in expected:
            with assert_raises(expected['error']):
                PowerSpace(harmonic_partner=harmonic_partner,
                           distribution_strategy=distribution_strategy,
                           logarithmic=logarithmic, nbin=nbin,
                           binbounds=binbounds)
        else:
            p = PowerSpace(harmonic_partner=harmonic_partner,
                           distribution_strategy=distribution_strategy,
                           logarithmic=logarithmic, nbin=nbin,
                           binbounds=binbounds)
            for key, value in expected.iteritems():
                if isinstance(value, np.ndarray):
                    assert_almost_equal(getattr(p, key), value)
                else:
                    assert_equal(getattr(p, key), value)

    @expand(get_distance_array_configs())
    def test_distance_array(self, harmonic_partner, expected):
        p = PowerSpace(harmonic_partner=harmonic_partner)
        assert_almost_equal(p.get_distance_array('not'), expected)

    @expand(get_weight_configs())
    def test_weight(self, harmonic_partner, x, power, axes,
                    inplace, expected):
        p = PowerSpace(harmonic_partner=harmonic_partner)
        res = p.weight(x, power, axes, inplace)
        assert_almost_equal(res, expected)
        if inplace:
            assert_(x is res)
