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
                  FieldArray

from d2o import distributed_data_object,\
                STRATEGIES

from test.common import expand

np.random.seed(123)

SPACES = [RGSpace((4,)), RGSpace((5))]
SPACE_COMBINATIONS = [(), SPACES[0], SPACES[1], SPACES]


class Test_Interface(unittest.TestCase):
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

    def test_hermitian_decomposition2(self):
        s1=(25,2)
        s2=(16,)
        ax1=((0,1,2),)
        ax2=((0,1),(2,))
        r2 = RGSpace(s1+s2, harmonic=True)
        ra = RGSpace(s1, harmonic=True)
        rb = RGSpace(s2, harmonic=True)
        v = np.empty(s1+s2,dtype=np.complex128)
        v.real = np.random.random(s1+s2)
        v.imag = np.random.random(s1+s2)
        f1=Field(r2,val=v,copy=True)
        f2=Field((ra,rb),val=v,copy=True)
        h2,a2 = Field._hermitian_decomposition((RGSpace(s1, harmonic=True),
                RGSpace(s2, harmonic=True)),f2.val,(0,1),ax2,False)
        h1,a1 = Field._hermitian_decomposition((RGSpace(s1+s2, harmonic=True),),
                f1.val,(0,),ax1,False)
        assert(np.max(np.abs(h1-h2))<1e-10)
        assert(np.max(np.abs(a1-a2))<1e-10)

#class Test_Initialization(unittest.TestCase):
#
#    @parameterized.expand(
#        itertools.product(SPACE_COMBINATIONS,
#                          []
#                          )
#    def test_
