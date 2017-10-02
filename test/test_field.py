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
from numpy.testing import assert_equal,\
                          assert_allclose
from nose.plugins.skip import SkipTest

from itertools import product

from nifty2go import Field,\
                  RGSpace,\
                  LMSpace,\
                  PowerSpace,\
                  DomainTuple,\
                  DiagonalOperator

from nifty2go.sugar import create_power_operator

from test.common import expand


SPACES = [RGSpace((4,)), RGSpace((5))]
SPACE_COMBINATIONS = [(), SPACES[0], SPACES[1], SPACES]


class Test_Interface(unittest.TestCase):
    @expand(product(SPACE_COMBINATIONS,
                    [['domain', DomainTuple],
                     ['val', np.ndarray],
                     ['shape', tuple],
                     ['dim', (np.int, np.int64)]]))
    def test_return_types(self, domain, attribute_desired_type):
        attribute = attribute_desired_type[0]
        desired_type = attribute_desired_type[1]
        f = Field(domain=domain)
        assert_equal(isinstance(getattr(f, attribute), desired_type), True)


class Test_Functionality(unittest.TestCase):
    @expand(product([RGSpace((8,), harmonic=True),
                     RGSpace((8, 8), harmonic=True, distances=0.123)],
                    [RGSpace((8,), harmonic=True),
                     LMSpace(12)]))
    def test_power_synthesize_analyze(self, space1, space2):
        np.random.seed(11)

        p1 = PowerSpace(space1)
        spec1 = lambda k: 42/(1+k)**2
        fp1 = Field(p1, val=spec1(p1.k_lengths))

        p2 = PowerSpace(space2)
        spec2 = lambda k: 42/(1+k)**3
        fp2 = Field(p2, val=spec2(p2.k_lengths))

        outer = np.outer(fp1.val, fp2.val)
        fp = Field((p1, p2), val=outer)

        samples = 500
        ps1 = 0.
        ps2 = 0.
        for ii in range(samples):
            sk = fp.power_synthesize(spaces=(0, 1), real_signal=True)

            sp = sk.power_analyze(spaces=(0, 1), keep_phase_information=False)
            ps1 += sp.sum(spaces=1)/fp2.sum()
            ps2 += sp.sum(spaces=0)/fp1.sum()

        assert_allclose(ps1.val/samples, fp1.val, rtol=0.2)
        assert_allclose(ps2.val/samples, fp2.val, rtol=0.2)


    @expand(product([RGSpace((8,), harmonic=True),
                     RGSpace((8, 8), harmonic=True, distances=0.123)],
                    [RGSpace((8,), harmonic=True),
                     LMSpace(12)]))
    def test_DiagonalOperator_power_analyze(self, space1, space2):
        np.random.seed(11)

        fulldomain = DomainTuple.make((space1, space2))

        p1 = PowerSpace(space1)
        spec1 = lambda k: 42/(1+k)**2
        fp1 = Field(p1, val=spec1(p1.k_lengths))

        p2 = PowerSpace(space2)
        spec2 = lambda k: 42/(1+k)**3
        fp2 = Field(p2, val=spec2(p2.k_lengths))

        S_1 = create_power_operator(space1, lambda x: np.sqrt(spec1(x)))
        S_2 = create_power_operator(space2, lambda x: np.sqrt(spec2(x)))
        S_1 = DiagonalOperator(S_1.diagonal().weight(-1),domain=fulldomain,
                               spaces=0)
        S_2 = DiagonalOperator(S_2.diagonal().weight(-1),domain=fulldomain,
                               spaces=1)

        samples = 500
        ps1 = 0.
        ps2 = 0.

        for ii in range(samples):
            rand_k = Field.from_random('normal', domain=fulldomain)
            sk = S_1.times(S_2.times(rand_k))
            sp = sk.power_analyze(spaces=(0, 1), keep_phase_information=False)
            ps1 += sp.sum(spaces=1)/fp2.sum()
            ps2 += sp.sum(spaces=0)/fp1.sum()

        assert_allclose(ps1.val/samples, fp1.val, rtol=0.2)
        assert_allclose(ps2.val/samples, fp2.val, rtol=0.2)

    def test_vdot(self):
        s=RGSpace((10,))
        f1=Field.from_random("normal",domain=s,dtype=np.complex128)
        f2=Field.from_random("normal",domain=s,dtype=np.complex128)
        assert_allclose(f1.vdot(f2),f1.vdot(f2,spaces=0))
        assert_allclose(f1.vdot(f2),np.conj(f2.vdot(f1)))


