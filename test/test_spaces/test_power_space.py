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

from itertools import chain, product

import nifty8 as ift
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises

from ..common import setup_function, teardown_function

pmp = pytest.mark.parametrize

HARMONIC_SPACES = [
    ift.RGSpace((8,), harmonic=True),
    ift.RGSpace((7, 8), harmonic=True),
    ift.RGSpace((6, 6), harmonic=True),
    ift.RGSpace((5, 5), harmonic=True),
    ift.RGSpace((4, 5, 7), harmonic=True),
    ift.LMSpace(6),
    ift.LMSpace(9)
]

# Try all sensible kinds of combinations of spaces and binning parameters
CONSISTENCY_CONFIGS_IMPLICIT = product(HARMONIC_SPACES, [None], [None, 3, 4],
                                       [True, False])
CONSISTENCY_CONFIGS_EXPLICIT = product(HARMONIC_SPACES, [[0., 1.3]], [None],
                                       [None])
CONSISTENCY_CONFIGS = chain(CONSISTENCY_CONFIGS_IMPLICIT,
                            CONSISTENCY_CONFIGS_EXPLICIT)

# [harmonic_partner, logarithmic, nbin, binbounds, expected]
CONSTRUCTOR_CONFIGS = [
    [1, False, None, None, {
        'error': (ValueError, NotImplementedError)
    }],
    [ift.RGSpace((8,)), False, None, None, {
        'error': ValueError
    }],
    [
        ift.RGSpace((8,), harmonic=True), None, None, None, {
            'harmonic':
            False,
            'shape': (5,),
            'size':
            5,
            'harmonic_partner':
            ift.RGSpace((8,), harmonic=True),
            'binbounds':
            None,
            'pindex':
            np.array([0, 1, 2, 3, 4, 3, 2, 1]),
            'k_lengths':
            np.array([0., 1., 2., 3., 4.]),
        }
    ],
    [
        ift.RGSpace((8,), harmonic=True), True, None, None, {
            'harmonic':
            False,
            'shape': (4,),
            'size':
            4,
            'harmonic_partner':
            ift.RGSpace((8,), harmonic=True),
            'binbounds': (0.5, 1.3228756555322954, 3.5),
            'pindex':
            np.array([0, 1, 2, 2, 3, 2, 2, 1]),
            'k_lengths':
            np.array([0., 1., 2.5, 4.]),
        }
    ],
]


def k_lengths_configs():
    da_0 = np.array([0, 1.0, 1.41421356, 2., 2.23606798, 2.82842712])
    return [
        [ift.RGSpace((4, 4), harmonic=True), da_0],
    ]


@pmp('attribute, expected_type', [
    ['harmonic_partner', ift.StructuredDomain],
    ['binbounds', type(None)],
    ['pindex', np.ndarray],
    ['k_lengths', np.ndarray],
])
def test_property_ret_type(attribute, expected_type):
    r = ift.RGSpace((4, 4), harmonic=True)
    p = ift.PowerSpace(r)
    ift.myassert(isinstance(getattr(p, attribute), expected_type))


@pmp('harmonic_partner, binbounds, nbin, logarithmic', CONSISTENCY_CONFIGS)
def test_rhopindexConsistency(harmonic_partner, binbounds, nbin, logarithmic):
    bb = ift.PowerSpace.useful_binbounds(harmonic_partner, logarithmic, nbin)
    p = ift.PowerSpace(harmonic_partner=harmonic_partner, binbounds=bb)

    assert_equal(
        np.bincount(p.pindex.ravel()),
        p.dvol,
        err_msg='rho is not equal to pindex degeneracy')


@pmp('harmonic_partner, logarithmic, nbin, binbounds, expected',
     CONSTRUCTOR_CONFIGS)
def test_constructor(harmonic_partner, logarithmic, nbin, binbounds, expected):
    if 'error' in expected:
        with assert_raises(expected['error']):
            bb = ift.PowerSpace.useful_binbounds(harmonic_partner, logarithmic,
                                                 nbin)
            ift.PowerSpace(harmonic_partner=harmonic_partner, binbounds=bb)
    else:
        bb = ift.PowerSpace.useful_binbounds(harmonic_partner, logarithmic,
                                             nbin)
        p = ift.PowerSpace(harmonic_partner=harmonic_partner, binbounds=bb)
        for key, value in expected.items():
            if isinstance(value, np.ndarray):
                assert_allclose(getattr(p, key), value)
            else:
                assert_equal(getattr(p, key), value)


@pmp('harmonic_partner, expected', k_lengths_configs())
def test_k_lengths(harmonic_partner, expected):
    p = ift.PowerSpace(harmonic_partner=harmonic_partner)
    assert_allclose(p.k_lengths, expected)


def test_dvol():
    hp = ift.RGSpace(10, harmonic=True)
    p = ift.PowerSpace(harmonic_partner=hp)
    v1 = hp.dvol
    v1 = hp.size*v1 if np.isscalar(v1) else np.sum(v1)
    v2 = p.dvol
    v2 = p.size*v2 if np.isscalar(v2) else np.sum(v2)
    assert_allclose(v1, v2)
