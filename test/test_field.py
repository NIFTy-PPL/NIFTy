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
                          assert_equal

from itertools import product

from nifty import Field,\
                  RGSpace,\
                  HPSpace,\
                  GLSpace

from d2o import distributed_data_object,\
                STRATEGIES

from test.common import expand

np.random.seed(123)

#  Exploring the parameter space
SPACES = [HPSpace(nside=1), RGSpace((4,)), GLSpace(nlat=2, nlon=3)]
SPACE_COMBINATIONS = [(), SPACES[0], SPACES[1], SPACES[2], SPACES[0:2], SPACES[1:]]
DTYPE_COMBINATIONS = [np.float64, np.complex128]
DIST_COMBINATIONS = STRATEGIES['global']
COPY_COMBINATIONS = [True, False]
VAL_COMBINATIONS = [np.empty(0), np.empty(0, dtype=np.complex128),
                    np.random.rand(12),  np.random.rand(12) + np.random.rand(12)*1j,
                    np.random.rand(4),  np.random.rand(4) + np.random.rand(4)*1j,
                    np.random.rand(6),  np.random.rand(6) + np.random.rand(6)*1j,
                    np.random.rand(48).reshape((12, 4)),
                    np.random.rand(48).reshape((12, 4)) + np.random.rand(48).reshape((12, 4))*1j,
                    np.random.rand(24).reshape((4, 6)),
                    np.random.rand(24).reshape((4, 6)) + np.random.rand(24).reshape((4, 6)) * 1j
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


def get_cast_configs():
    keys = ('__class__', 'shape', 'distribution_strategy', 'dtype')
    values = product([distributed_data_object], [(), (12,), (4,), (6,), (12, 4), (4, 6)], ['not'],
                     [np.float, np.complex])
    output = [dict(zip(keys, l)) for l in values]
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

if __name__ == '__main__':
    unittest.main()


'''
    @expand(product(FIELD_COMBINATIONS, EXPECTED_FUNCTIONALITY['power_synthesize']))
    def test_power_synthesize(self, field, expected_functionality):
        pow_syn = field.power_synthesize()
        for key, value in expected_functionality:
            assert_equal(getattr(pow_syn, key), value)

    @expand(product(SPACE_COMBINATIONS, EXPECTED_FUNCTIONALITY['power_analyze']))
    def test_power_analyze(self, field, expected_functionality):
        pow_an = field.power_analyze()
        for key, value in expected_functionality:
            assert_equal(getattr(pow_an, key), value)

    @expand(product(SPACE_COMBINATIONS, EXPECTED_FUNCTIONALITY['weight']))
    def test_weight(self, field, expected_functionality):
        weight = field.power_synthesize()
        for key, value in expected_functionality:
            assert_equal(getattr(weight, key), value)

    @expand(product(SPACE_COMBINATIONS, EXPECTED_FUNCTIONALITY['dot']))
    def test_dot(self, field, expected_functionality):
        dot = field.dot()
        for key, value in expected_functionality:
            assert_equal(getattr(dot, key), value)

    @expand(product(SPACE_COMBINATIONS, EXPECTED_FUNCTIONALITY['norm']))
    def test_norm(self, field, expected_functionality):
        norm = field.norm()e
        for key, value in expected_functionality:
            assert_equal(getattr(norm, key), value)

    @expand(product(SPACE_COMBINATIONS, EXPECTED_FUNCTIONALITY['conjugate']))
    def test_conjugate(self, field, expected_functionality):
        conjugate = field.conjugate()
        for key, value in expected_functionality:
            assert_equal(getattr(conjugate, key), value)

'''