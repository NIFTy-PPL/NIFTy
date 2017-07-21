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
                          assert_almost_equal,\
                          assert_allclose

from itertools import product

from nifty import Field,\
                  RGSpace,\
                  LMSpace,\
                  PowerSpace

from d2o import distributed_data_object

from test.common import expand


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


class Test_Functionality(unittest.TestCase):
    @expand(product([True, False], [True, False],
                    [True, False], [True, False],
                    [(1,), (4,), (5,)], [(1,), (6,), (7,)]))
    def test_hermitian_decomposition(self, z1, z2, preserve, complexdata,
                                     s1, s2):
        np.random.seed(123)
        r1 = RGSpace(s1, harmonic=True, zerocenter=(z1,))
        r2 = RGSpace(s2, harmonic=True, zerocenter=(z2,))
        ra = RGSpace(s1+s2, harmonic=True, zerocenter=(z1, z2))

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

        assert_almost_equal(h1.get_full_data(), h2.get_full_data())
        assert_almost_equal(a1.get_full_data(), a2.get_full_data())
        assert_almost_equal(h1.get_full_data(), h3.get_full_data())
        assert_almost_equal(a1.get_full_data(), a3.get_full_data())

    @expand(product([RGSpace((8,), harmonic=True,
                             zerocenter=False),
                     RGSpace((8, 8), harmonic=True, distances=0.123,
                             zerocenter=True)],
                    [RGSpace((8,), harmonic=True,
                             zerocenter=False),
                     LMSpace(12)]))
    def test_power_synthesize_analyze(self, space1, space2):
        p1 = PowerSpace(space1)
        spec1 = lambda k: 42/(1+k)**2
        fp1 = Field(p1, val=spec1)

        p2 = PowerSpace(space2)
        spec2 = lambda k: 42/(1+k)**3
        fp2 = Field(p2, val=spec2)

        outer = np.outer(fp1.val.get_full_data(), fp2.val.get_full_data())
        fp = Field((p1, p2), val=outer)

        samples = 1000
        ps1 = 0.
        ps2 = 0.
        for ii in xrange(samples):
            sk = fp.power_synthesize(spaces=(0, 1), real_signal=True)

            sp = sk.power_analyze(spaces=(0, 1), keep_phase_information=False)
            ps1 += sp.sum(spaces=1)/fp2.sum()
            ps2 += sp.sum(spaces=0)/fp1.sum()

        assert_allclose(ps1.val.get_full_data()/samples,
                        fp1.val.get_full_data(),
                        rtol=0.1)
        assert_allclose(ps2.val.get_full_data()/samples,
                        fp2.val.get_full_data(),
                        rtol=0.1)

    def test_vdot(self):
        s=RGSpace((10,))
        f1=Field.from_random("normal",domain=s,dtype=np.complex128)
        f2=Field.from_random("normal",domain=s,dtype=np.complex128)
        assert_allclose(f1.vdot(f2),f1.vdot(f2,spaces=0))
        assert_allclose(f1.vdot(f2),np.conj(f2.vdot(f1)))


