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
from numpy.testing import assert_, assert_equal, assert_allclose,\
        assert_raises
import nifty4 as ift
from test.common import expand
from itertools import product, chain

HARMONIC_SPACES = [ift.RGSpace((8,), harmonic=True),
                   ift.RGSpace((7, 8), harmonic=True),
                   ift.RGSpace((6, 6), harmonic=True),
                   ift.RGSpace((5, 5), harmonic=True),
                   ift.RGSpace((4, 5, 7), harmonic=True),
                   ift.LMSpace(6),
                   ift.LMSpace(9)]


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
    [ift.RGSpace((8,)), False, None, None, {'error': ValueError}],
    [ift.RGSpace((8,), harmonic=True), None, None, None, {
        'harmonic': False,
        'shape': (5,),
        'dim': 5,
        'harmonic_partner': ift.RGSpace((8,), harmonic=True),
        'binbounds': None,
        'pindex': ift.dobj.from_global_data(
            np.array([0, 1, 2, 3, 4, 3, 2, 1])),
        'k_lengths': np.array([0., 1., 2., 3., 4.]),
        }],
    [ift.RGSpace((8,), harmonic=True), True, None, None, {
        'harmonic': False,
        'shape': (4,),
        'dim': 4,
        'harmonic_partner': ift.RGSpace((8,), harmonic=True),
        'binbounds': (0.5, 1.3228756555322954, 3.5),
        'pindex': ift.dobj.from_global_data(
            np.array([0, 1, 2, 2, 3, 2, 2, 1])),
        'k_lengths': np.array([0., 1., 2.5, 4.]),
        }],
    ]


def k_lengths_configs():
    da_0 = np.array([0, 1.0, 1.41421356, 2., 2.23606798, 2.82842712])
    return [
        [ift.RGSpace((4, 4), harmonic=True),  da_0],
        ]


class PowerSpaceInterfaceTest(unittest.TestCase):
    @expand([
        ['harmonic_partner', ift.Space],
        ['binbounds', type(None)],
        ['pindex', ift.dobj.data_object],
        ['k_lengths', np.ndarray],
        ])
    def test_property_ret_type(self, attribute, expected_type):
        r = ift.RGSpace((4, 4), harmonic=True)
        p = ift.PowerSpace(r)
        assert_(isinstance(getattr(p, attribute), expected_type))


class PowerSpaceConsistencyCheck(unittest.TestCase):
    @expand(CONSISTENCY_CONFIGS)
    def test_rhopindexConsistency(self, harmonic_partner,
                                  binbounds, nbin, logarithmic):
        bb = ift.PowerSpace.useful_binbounds(harmonic_partner, logarithmic,
                                             nbin)
        p = ift.PowerSpace(harmonic_partner=harmonic_partner, binbounds=bb)

        assert_equal(np.bincount(ift.dobj.to_global_data(p.pindex).ravel()),
                     p.dvol(), err_msg='rho is not equal to pindex degeneracy')


class PowerSpaceFunctionalityTest(unittest.TestCase):
    @expand(CONSTRUCTOR_CONFIGS)
    def test_constructor(self, harmonic_partner,
                         logarithmic, nbin, binbounds, expected):
        if 'error' in expected:
            with assert_raises(expected['error']):
                bb = ift.PowerSpace.useful_binbounds(harmonic_partner,
                                                     logarithmic, nbin)
                ift.PowerSpace(harmonic_partner=harmonic_partner, binbounds=bb)
        else:
            bb = ift.PowerSpace.useful_binbounds(harmonic_partner,
                                                 logarithmic, nbin)
            p = ift.PowerSpace(harmonic_partner=harmonic_partner, binbounds=bb)
            for key, value in expected.items():
                if isinstance(value, np.ndarray):
                    assert_allclose(getattr(p, key), value)
                else:
                    assert_equal(getattr(p, key), value)

    @expand(k_lengths_configs())
    def test_k_lengths(self, harmonic_partner, expected):
        p = ift.PowerSpace(harmonic_partner=harmonic_partner)
        assert_allclose(p.k_lengths, expected)

    def test_dvol(self):
        hp = ift.RGSpace(10, harmonic=True)
        p = ift.PowerSpace(harmonic_partner=hp)
        v1 = hp.dvol()
        v1 = hp.dim*v1 if np.isscalar(v1) else np.sum(v1)
        v2 = p.dvol()
        v2 = p.dim*v2 if np.isscalar(v2) else np.sum(v2)
        assert_allclose(v1, v2)
