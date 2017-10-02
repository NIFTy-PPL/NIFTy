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

from numpy.testing import assert_, assert_equal, assert_almost_equal,\
        assert_raises
from nifty2go import PowerSpace, RGSpace, Space, LMSpace
from test.common import expand
from itertools import product, chain

HARMONIC_SPACES = [RGSpace((8,), harmonic=True),
                   RGSpace((7, 8), harmonic=True),
                   RGSpace((6, 6), harmonic=True),
                   RGSpace((5, 5), harmonic=True),
                   RGSpace((4, 5, 7), harmonic=True),
                   LMSpace(6),
                   LMSpace(9)]


# Try all sensible kinds of combinations of spaces and binning parameters
CONSISTENCY_CONFIGS_IMPLICIT = product(HARMONIC_SPACES,
                                       [None], [None, 3, 4], [True, False])
CONSISTENCY_CONFIGS_EXPLICIT = product(HARMONIC_SPACES,
                                       [[0., 1.3]], [None], [None])
CONSISTENCY_CONFIGS = chain(CONSISTENCY_CONFIGS_IMPLICIT,
                            CONSISTENCY_CONFIGS_EXPLICIT)

# [harmonic_partner, logarithmic, nbin, binbounds, expected]
CONSTRUCTOR_CONFIGS = [
    [1, False, None, None, {'error': (ValueError, NotImplementedError)}],
    [RGSpace((8,)), False, None, None, {'error': ValueError}],
    [RGSpace((8,), harmonic=True), None, None, None, {
        'harmonic': True,
        'shape': (5,),
        'dim': 5,
        'harmonic_partner': RGSpace((8,), harmonic=True),
        'binbounds': None,
        'pindex': np.array([0, 1, 2, 3, 4, 3, 2, 1]),
        'kindex': np.array([0., 1., 2., 3., 4.]),
        'rho': np.array([1, 2, 2, 2, 1]),
        }],
    [RGSpace((8,), harmonic=True), True, None, None, {
        'harmonic': True,
        'shape': (4,),
        'dim': 4,
        'harmonic_partner': RGSpace((8,), harmonic=True),
        'binbounds': (0.5, 1.3228756555322954, 3.5),
        'pindex': np.array([0, 1, 2, 2, 3, 2, 2, 1]),
        'kindex': np.array([0., 1., 2.5, 4.]),
        'rho': np.array([1, 2, 4, 1]),
        }],
    ]


def get_k_length_array_configs():
    da_0 = np.array([0, 1.0, 1.41421356, 2., 2.23606798, 2.82842712])
    return [
        [RGSpace((4, 4), harmonic=True),  da_0],
        ]


class PowerSpaceInterfaceTest(unittest.TestCase):
    @expand([
        ['harmonic_partner', Space],
        ['binbounds', type(None)],
        ['pindex', np.ndarray],
        ['kindex', np.ndarray],
        ['rho', np.ndarray],
        ])
    def test_property_ret_type(self, attribute, expected_type):
        r = RGSpace((4, 4), harmonic=True)
        p = PowerSpace(r)
        assert_(isinstance(getattr(p, attribute), expected_type))


class PowerSpaceConsistencyCheck(unittest.TestCase):
    @expand(CONSISTENCY_CONFIGS)
    def test_rhopindexConsistency(self, harmonic_partner,
                                  binbounds, nbin, logarithmic):
        bb = PowerSpace.useful_binbounds(harmonic_partner, logarithmic, nbin)
        p = PowerSpace(harmonic_partner=harmonic_partner, binbounds=bb)

        assert_equal(np.bincount(p.pindex.ravel()), p.rho,
                     err_msg='rho is not equal to pindex degeneracy')


class PowerSpaceFunctionalityTest(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, harmonic_partner,
                         logarithmic, nbin, binbounds, expected):
        if 'error' in expected:
            with assert_raises(expected['error']):
                bb = PowerSpace.useful_binbounds(harmonic_partner,
                                                 logarithmic, nbin)
                PowerSpace(harmonic_partner=harmonic_partner, binbounds=bb)
        else:
            bb = PowerSpace.useful_binbounds(harmonic_partner,
                                             logarithmic, nbin)
            p = PowerSpace(harmonic_partner=harmonic_partner, binbounds=bb)
            for key, value in expected.items():
                if isinstance(value, np.ndarray):
                    assert_almost_equal(getattr(p, key), value)
                else:
                    assert_equal(getattr(p, key), value)

    @expand(get_k_length_array_configs())
    def test_k_length_array(self, harmonic_partner, expected):
        p = PowerSpace(harmonic_partner=harmonic_partner)
        assert_almost_equal(p.get_k_length_array(), expected)

    def test_dvol(self):
        p = PowerSpace(harmonic_partner=RGSpace(10,harmonic=True))
        assert_almost_equal(p.dvol(),p.rho)
