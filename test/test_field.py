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

import unittest

import numpy as np
from numpy.testing import assert_,\
                          assert_allclose,\
                          assert_equal

from nose.plugins.skip import SkipTest

from itertools import product

from nifty import Field,\
                  RGSpace,\
                  HPSpace,\
                  GLSpace,\
                  LMSpace,\
                  PowerSpace,\
                  nifty_configuration

import d2o
from d2o import distributed_data_object, STRATEGIES

from test.common import expand


#  Exploring the parameter space
SPACES = [HPSpace(nside=1), RGSpace((4,)), GLSpace(nlat=2, nlon=3)]
SPACE_COMBINATIONS = [(), SPACES[0], SPACES[1], SPACES[2], SPACES[0:2], SPACES[1:]]
DTYPE_COMBINATIONS = [np.float64, np.complex128]
DIST_COMBINATIONS = STRATEGIES['global']
COPY_COMBINATIONS = [True, False]
VAL_COMBINATIONS = [None, None,
                    np.random.randn(12),  np.random.randn(12) + np.random.randn(12)*1j,
                    np.random.randn(4),  np.random.randn(4) + np.random.randn(4)*1j,
                    np.random.randn(6),  np.random.randn(6) + np.random.randn(6)*1j,
                    np.random.randn(48).reshape((12, 4)),
                    np.random.randn(48).reshape((12, 4)) + np.random.randn(48).reshape((12, 4))*1j,
                    np.random.randn(24).reshape((4, 6)),
                    np.random.randn(24).reshape((4, 6)) + np.random.randn(24).reshape((4, 6)) * 1j
                    ]

# The expected properties
EXPECTED_PROPERTIES = [{'shape': (),  # empty, real
                        'domain_axes': (),
                        'dof': 0,
                        'dim': 0,
                        'total_volume': 0.0,
                        },
                       {'shape': (),  # empty, complex
                        'domain_axes': (),
                        'dof': 0,
                        'dim': 0,
                        'total_volume': 0.0,
                        },
                       {'shape': (12,),  # hp, real
                        'domain_axes': ((0,),),
                        'dof': 12,
                        'dim': 12,
                        'total_volume': 4*np.pi,
                        },
                       {'shape': (12,),  # hp, complex
                        'domain_axes': ((0,),),
                        'dof': 24,
                        'dim': 12,
                        'total_volume': 4*np.pi,
                        },
                       {'shape': (4,),  # rg, real
                        'domain_axes': ((0,),),
                        'dof': 4,
                        'dim': 4,
                        'total_volume': 1.0,
                        },
                       {'shape': (4,),   # rg, complex
                        'domain_axes': ((0,),),
                        'dof': 8,
                        'dim': 4,
                        'total_volume': 1.0,
                        },
                       {'shape': (6,),  # gl, real
                        'domain_axes': ((0,),),
                        'dof': 6,
                        'dim': 6,
                        'total_volume': 4 * np.pi,
                        },
                       {'shape': (6,),  # gl, complex
                        'domain_axes': ((0,),),
                        'dof': 12,
                        'dim': 6,
                        'total_volume': 4 * np.pi,
                        },
                       {'shape': (12, 4,),  # prod(hp,rg), real
                        'domain_axes': ((0,), (1,)),
                        'dof': 48,
                        'dim': 48,
                        'total_volume':  4 * np.pi,
                        },
                       {'shape': (12, 4,),  # prod(hp,rg), complex
                        'domain_axes': ((0,), (1,)),
                        'dof': 96,
                        'dim': 48,
                        'total_volume': 4 * np.pi,
                        },
                       {'shape': (4, 6,),  # prod(rg,gl), real
                        'domain_axes': ((0,), (1,)),
                        'dof': 24,
                        'dim': 24,
                        'total_volume': 4 * np.pi,
                        },
                       {'shape': (4, 6,),  # prod(rg,gl), complex
                        'domain_axes': ((0,), (1,)),
                        'dof': 48,
                        'dim': 24,
                        'total_volume': 4 * np.pi,
                        },
                       ]


# The expected output of the methods

# cast test config, testing for shape, dtype and dist-strategy of the resulting d2o object,
# dist-strategy combinatorics maybe expanded
def get_cast_configs():
    keys = ('shape', 'distribution_strategy', 'dtype')
    values = product([(), (12,), (4,), (6,), (12, 4), (4, 6)], ['not'],
                     [np.float, np.complex])
    output = [dict(zip(keys, l)) for l in values]
    return output


# pnorm test config for  p=1
def get_norm_configs():
    output = []
    val = np.copy(VAL_COMBINATIONS)
    val[0] = np.zeros(1)
    val[1] = np.zeros(1)
    for i in val:
        output.append([np.sqrt(sum(abs(i.flatten())**2))])
    print output
    return output


# The tests
class FieldInterfaceTest(unittest.TestCase):
    @expand(product(SPACE_COMBINATIONS,
                    [['distribution_strategy', str],
                     ['domain', tuple],
                     ['domain_axes', tuple],
                     ['val', distributed_data_object],
                     ['shape', tuple],
                     ['dim', np.int],
                     ['dof', np.int],
                     ['total_volume', np.float]]))
    def test_return_types(self, domain, attribute_desired_type):
        attribute = attribute_desired_type[0]
        desired_type = attribute_desired_type[1]
        f = Field(domain=domain)
        assert_(isinstance(getattr(f, attribute), desired_type))


class FieldPropertyCheck(unittest.TestCase):
    @expand(zip(product(SPACE_COMBINATIONS, DTYPE_COMBINATIONS), EXPECTED_PROPERTIES))
    def test_consistency(self, setters, expected_attributes):
        f = Field(domain=setters[0], dtype=setters[1])
        for key, value in expected_attributes.iteritems():
            assert_equal(getattr(f, key), value)


class FieldFunctionalityTest(unittest.TestCase):
    @expand(zip(zip(product(SPACE_COMBINATIONS, DTYPE_COMBINATIONS), VAL_COMBINATIONS), 12*['not'], get_cast_configs()))
    def test_cast(self, setters, dist, expected_functionality):
        field = Field(domain=setters[0][0], val=setters[1], distribution_strategy=dist, dtype=setters[0][1])
        cast = field.cast()
        for key, value in expected_functionality.iteritems():
            assert_equal(getattr(cast, key), value)

#  not yet working
#    @expand(zip(product(SPACE_COMBINATIONS, DTYPE_COMBINATIONS), VAL_COMBINATIONS, get_norm_configs()))
#    def test_norm(self, setters, values, expected_functionality):
#        field = Field(domain=setters[0], val=values,  dtype=setters[1])
#        norm = field.norm()
#        for value in expected_functionality:
#            assert_equal(norm, value)


class Test_Functionality(unittest.TestCase):
    @expand(product([True, False], [True, False],
                    [True, False], [True, False],
                    [(1,), (4,), (5,)], [(1,), (6,), (7,)]))
    def test_hermitian_decomposition(self, z1, z2, preserve, complexdata,
                                     s1, s2):
        try:
            r1 = RGSpace(s1, harmonic=True, zerocenter=(z1,))
            r2 = RGSpace(s2, harmonic=True, zerocenter=(z2,))
            ra = RGSpace(s1+s2, harmonic=True, zerocenter=(z1, z2))
        except ValueError:
            raise SkipTest

        if preserve:
            complexdata=True
        v = np.random.random(s1+s2)
        if complexdata:
            v = v + 1j*np.random.random(s1+s2)
        f1 = Field(ra, val=v, copy=True)
        f2 = Field((r1, r2), val=v, copy=True)
        h1, a1 = Field._hermitian_decomposition((ra,), f1.val, (0,),
                                                ((0, 1,),), preserve)
        h2, a2 = Field._hermitian_decomposition((r1, r2), f2.val, (0, 1),
                                                ((0,), (1,)), preserve)
        h3, a3 = Field._hermitian_decomposition((r1, r2), f2.val, (1, 0),
                                                ((0,), (1,)), preserve)

        assert_allclose(h1.get_full_data(), h2.get_full_data())
        assert_allclose(a1.get_full_data(), a2.get_full_data())
        assert_allclose(h1.get_full_data(), h3.get_full_data())
        assert_allclose(a1.get_full_data(), a3.get_full_data())

    @expand(product([RGSpace((8,), harmonic=True,
                             zerocenter=False),
                     RGSpace((8, 8), harmonic=True, distances=0.123,
                             zerocenter=True)],
                    [RGSpace((8,), harmonic=True,
                             zerocenter=False),
                     LMSpace(12)],
                    ['real', 'complex']))
    def test_power_synthesize_analyze(self, space1, space2, base):
        nifty_configuration['harmonic_rg_base'] = base

        d2o.random.seed(11)

        p1 = PowerSpace(space1)
        spec1 = lambda k: 42/(1+k)**2
        fp1 = Field(p1, val=spec1)

        p2 = PowerSpace(space2)
        spec2 = lambda k: 42/(1+k)**3
        fp2 = Field(p2, val=spec2)

        outer = np.outer(fp1.val.get_full_data(), fp2.val.get_full_data())
        fp = Field((p1, p2), val=outer)

        samples = 2000
        ps1 = 0.
        ps2 = 0.
        for ii in xrange(samples):
            sk = fp.power_synthesize(spaces=(0, 1), real_signal=True)

            sp = sk.power_analyze(spaces=(0, 1), keep_phase_information=False)
            ps1 += sp.sum(spaces=1)/fp2.sum()
            ps2 += sp.sum(spaces=0)/fp1.sum()

        assert_allclose(ps1.val.get_full_data()/samples,
                        fp1.val.get_full_data(),
                        rtol=0.2)
        assert_allclose(ps2.val.get_full_data()/samples,
                        fp2.val.get_full_data(),
                        rtol=0.2)

    def test_vdot(self):
        s=RGSpace((10,))
        f1=Field.from_random("normal",domain=s,dtype=np.complex128)
        f2=Field.from_random("normal",domain=s,dtype=np.complex128)
        assert_allclose(f1.vdot(f2),f1.vdot(f2,spaces=0))
        assert_allclose(f1.vdot(f2),np.conj(f2.vdot(f1)))
