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
from numpy.testing import assert_equal, assert_allclose
from itertools import product
import nifty2go as ift
from nifty2go import dobj
from test.common import expand


SPACES = [ift.RGSpace((4,)), ift.RGSpace((5))]
SPACE_COMBINATIONS = [(), SPACES[0], SPACES[1], SPACES]


class Test_Interface(unittest.TestCase):
    @expand(product(SPACE_COMBINATIONS,
                    [['domain', ift.DomainTuple],
                     ['val', ift.dobj.data_object],
                     ['shape', tuple],
                     ['dim', (np.int, np.int64)]]))
    def test_return_types(self, domain, attribute_desired_type):
        attribute = attribute_desired_type[0]
        desired_type = attribute_desired_type[1]
        f = ift.Field(domain=domain)
        assert_equal(isinstance(getattr(f, attribute), desired_type), True)


def _spec1(k):
    return 42/(1.+k)**2


def _spec2(k):
    return 42/(1.+k)**3


class Test_Functionality(unittest.TestCase):
    @expand(product([ift.RGSpace((8,), harmonic=True),
                     ift.RGSpace((8, 8), harmonic=True, distances=0.123)],
                    [ift.RGSpace((8,), harmonic=True),
                     ift.LMSpace(12)]))
    def test_power_synthesize_analyze(self, space1, space2):
        np.random.seed(11)

        p1 = ift.PowerSpace(space1)
        fp1 = ift.Field(p1, val=dobj.from_global_data(_spec1(p1.k_lengths)))

        p2 = ift.PowerSpace(space2)
        fp2 = ift.Field(p2, val=dobj.from_global_data(_spec2(p2.k_lengths)))

        outer = dobj.from_global_data(np.outer(dobj.to_global_data(fp1.val), dobj.to_global_data(fp2.val)))
        fp = ift.Field((p1, p2), val=outer)

        samples = 500
        ps1 = 0.
        ps2 = 0.
        for ii in range(samples):
            sk = ift.power_synthesize(fp, spaces=(0, 1), real_signal=True)

            sp = ift.power_analyze(sk, spaces=(0, 1),
                                   keep_phase_information=False)
            ps1 += sp.sum(spaces=1)/fp2.sum()
            ps2 += sp.sum(spaces=0)/fp1.sum()

        assert_allclose(dobj.to_global_data(ps1.val/samples), dobj.to_global_data(fp1.val), rtol=0.2)
        assert_allclose(dobj.to_global_data(ps2.val/samples), dobj.to_global_data(fp2.val), rtol=0.2)

    @expand(product([ift.RGSpace((8,), harmonic=True),
                     ift.RGSpace((8, 8), harmonic=True, distances=0.123)],
                    [ift.RGSpace((8,), harmonic=True),
                     ift.LMSpace(12)]))
    def test_DiagonalOperator_power_analyze(self, space1, space2):
        np.random.seed(11)

        fulldomain = ift.DomainTuple.make((space1, space2))

        p1 = ift.PowerSpace(space1)
        fp1 = ift.Field(p1, val=dobj.from_global_data(_spec1(p1.k_lengths)))

        p2 = ift.PowerSpace(space2)
        fp2 = ift.Field(p2, val=dobj.from_global_data(_spec2(p2.k_lengths)))

        S_1 = ift.create_power_field(space1, lambda x: np.sqrt(_spec1(x)))
        S_1 = ift.DiagonalOperator(S_1, domain=fulldomain, spaces=0)
        S_2 = ift.create_power_field(space2, lambda x: np.sqrt(_spec2(x)))
        S_2 = ift.DiagonalOperator(S_2, domain=fulldomain, spaces=1)

        samples = 500
        ps1 = 0.
        ps2 = 0.

        for ii in range(samples):
            rand_k = ift.Field.from_random('normal', domain=fulldomain)
            sk = S_1.times(S_2.times(rand_k))
            sp = ift.power_analyze(sk, spaces=(0, 1),
                                   keep_phase_information=False)
            ps1 += sp.sum(spaces=1)/fp2.sum()
            ps2 += sp.sum(spaces=0)/fp1.sum()

        assert_allclose(dobj.to_global_data(ps1.val/samples), dobj.to_global_data(fp1.val), rtol=0.2)
        assert_allclose(dobj.to_global_data(ps2.val/samples), dobj.to_global_data(fp2.val), rtol=0.2)

    def test_vdot(self):
        s = ift.RGSpace((10,))
        f1 = ift.Field.from_random("normal", domain=s, dtype=np.complex128)
        f2 = ift.Field.from_random("normal", domain=s, dtype=np.complex128)
        assert_allclose(f1.vdot(f2), f1.vdot(f2, spaces=0))
        assert_allclose(f1.vdot(f2), np.conj(f2.vdot(f1)))
