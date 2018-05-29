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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import unittest
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
from itertools import product
import nifty4 as ift
from test.common import expand


SPACES = [ift.RGSpace((4,)), ift.RGSpace((5))]
SPACE_COMBINATIONS = [(), SPACES[0], SPACES[1], SPACES]


class Test_Interface(unittest.TestCase):
    @expand(product(SPACE_COMBINATIONS,
                    [['domain', ift.DomainTuple],
                     ['val', ift.dobj.data_object],
                     ['shape', tuple],
                     ['size', (np.int, np.int64)]]))
    def test_return_types(self, domain, attribute_desired_type):
        attribute = attribute_desired_type[0]
        desired_type = attribute_desired_type[1]
        f = ift.Field.full(domain, 1.)
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
        fp1 = ift.PS_field(p1, _spec1)
        p2 = ift.PowerSpace(space2)
        fp2 = ift.PS_field(p2, _spec2)
        outer = np.outer(fp1.to_global_data(), fp2.to_global_data())
        fp = ift.Field.from_global_data((p1, p2), outer)

        op1 = ift.create_power_operator((space1, space2), _spec1, 0)
        op2 = ift.create_power_operator((space1, space2), _spec2, 1)
        opfull = op2*op1

        samples = 500
        sc1 = ift.StatCalculator()
        sc2 = ift.StatCalculator()
        for ii in range(samples):
            sk = opfull.draw_sample()

            sp = ift.power_analyze(sk, spaces=(0, 1),
                                   keep_phase_information=False)
            sc1.add(sp.sum(spaces=1)/fp2.sum())
            sc2.add(sp.sum(spaces=0)/fp1.sum())

        assert_allclose(sc1.mean.local_data, fp1.local_data, rtol=0.2)
        assert_allclose(sc2.mean.local_data, fp2.local_data, rtol=0.2)

    @expand(product([ift.RGSpace((8,), harmonic=True),
                     ift.RGSpace((8, 8), harmonic=True, distances=0.123)],
                    [ift.RGSpace((8,), harmonic=True),
                     ift.LMSpace(12)]))
    def test_DiagonalOperator_power_analyze2(self, space1, space2):
        np.random.seed(11)

        fp1 = ift.PS_field(ift.PowerSpace(space1), _spec1)
        fp2 = ift.PS_field(ift.PowerSpace(space2), _spec2)

        S_1 = ift.create_power_operator((space1, space2), _spec1, 0)
        S_2 = ift.create_power_operator((space1, space2), _spec2, 1)
        S_full = S_2*S_1

        samples = 500
        sc1 = ift.StatCalculator()
        sc2 = ift.StatCalculator()

        for ii in range(samples):
            sk = S_full.draw_sample()
            sp = ift.power_analyze(sk, spaces=(0, 1),
                                   keep_phase_information=False)
            sc1.add(sp.sum(spaces=1)/fp2.sum())
            sc2.add(sp.sum(spaces=0)/fp1.sum())

        assert_allclose(sc1.mean.local_data, fp1.local_data, rtol=0.2)
        assert_allclose(sc2.mean.local_data, fp2.local_data, rtol=0.2)

    def test_vdot(self):
        s = ift.RGSpace((10,))
        f1 = ift.Field.from_random("normal", domain=s, dtype=np.complex128)
        f2 = ift.Field.from_random("normal", domain=s, dtype=np.complex128)
        assert_allclose(f1.vdot(f2), f1.vdot(f2, spaces=0))
        assert_allclose(f1.vdot(f2), np.conj(f2.vdot(f1)))

    def test_vdot2(self):
        x1 = ift.RGSpace((200,))
        x2 = ift.RGSpace((150,))
        m = ift.Field.full((x1, x2), .5)
        res = m.vdot(m, spaces=1)
        assert_allclose(res.local_data, 37.5)

    def test_lock(self):
        s1 = ift.RGSpace((10,))
        f1 = ift.Field(s1, 27)
        assert_equal(f1.locked, False)
        f1.lock()
        assert_equal(f1.locked, True)
        with assert_raises(ValueError):
            f1 += f1
        assert_equal(f1.locked_copy() is f1, True)

    def test_fill(self):
        s1 = ift.RGSpace((10,))
        f1 = ift.Field(s1, 27)
        assert_equal(f1.fill(10).local_data, 10)

    def test_dataconv(self):
        s1 = ift.RGSpace((10,))
        ld = np.arange(ift.dobj.local_shape(s1.shape)[0])
        gd = np.arange(s1.shape[0])
        assert_equal(ld, ift.from_local_data(s1, ld).local_data)
        assert_equal(gd, ift.from_global_data(s1, gd).to_global_data())

    def test_cast_domain(self):
        s1 = ift.RGSpace((10,))
        s2 = ift.RGSpace((10,), distances=20.)
        d = np.arange(s1.shape[0])
        d2 = ift.from_global_data(s1, d).cast_domain(s2).to_global_data()
        assert_equal(d, d2)

    def test_empty_domain(self):
        f = ift.Field((), 5)
        assert_equal(f.to_global_data(), 5)
        f = ift.Field(None, 5)
        assert_equal(f.to_global_data(), 5)
        assert_equal(f.empty_copy().domain, f.domain)
        assert_equal(f.empty_copy().dtype, f.dtype)
        assert_equal(f.copy().domain, f.domain)
        assert_equal(f.copy().dtype, f.dtype)
        assert_equal(f.copy().local_data, f.local_data)
        assert_equal(f.copy() is f, False)

    def test_trivialities(self):
        s1 = ift.RGSpace((10,))
        f1 = ift.Field(s1, 27)
        assert_equal(f1.local_data, f1.real.local_data)
        f1 = ift.Field(s1, 27.+3j)
        assert_equal(f1.real.local_data, 27.)
        assert_equal(f1.imag.local_data, 3.)
        assert_equal(f1.local_data, +f1.local_data)
        assert_equal(f1.sum(), f1.sum(0))
        f1 = ift.from_global_data(s1, np.arange(10))
        assert_equal(f1.min(), 0)
        assert_equal(f1.max(), 9)
        assert_equal(f1.prod(), 0)

    def test_weight(self):
        s1 = ift.RGSpace((10,))
        f = ift.Field(s1, 10.)
        f2 = f.copy()
        f.weight(1, out=f2)
        assert_equal(f.weight(1).local_data, f2.local_data)
        assert_equal(f.total_volume(), 1)
        assert_equal(f.total_volume(0), 1)
        assert_equal(f.total_volume((0,)), 1)
        assert_equal(f.scalar_weight(), 0.1)
        assert_equal(f.scalar_weight(0), 0.1)
        assert_equal(f.scalar_weight((0,)), 0.1)
        s1 = ift.GLSpace(10)
        f = ift.Field(s1, 10.)
        assert_equal(f.scalar_weight(), None)
        assert_equal(f.scalar_weight(0), None)
        assert_equal(f.scalar_weight((0,)), None)

    @expand(product([ift.RGSpace(10), ift.GLSpace(10)],
                    [np.float64, np.complex128]))
    def test_reduction(self, dom, dt):
        s1 = ift.Field(dom, 1., dtype=dt)
        assert_allclose(s1.mean(), 1.)
        assert_allclose(s1.mean(0), 1.)
        assert_allclose(s1.var(), 0., atol=1e-14)
        assert_allclose(s1.var(0), 0., atol=1e-14)
        assert_allclose(s1.std(), 0., atol=1e-14)
        assert_allclose(s1.std(0), 0., atol=1e-14)

    def test_err(self):
        s1 = ift.RGSpace((10,))
        s2 = ift.RGSpace((11,))
        f1 = ift.Field(s1, 27)
        with assert_raises(ValueError):
            f2 = ift.Field(s2, f1)
        with assert_raises(ValueError):
            f2 = ift.Field(s2, f1.val)
        with assert_raises(TypeError):
            f2 = ift.Field(s2, "xyz")
        with assert_raises(TypeError):
            if f1:
                pass
        with assert_raises(TypeError):
            f1.full((2, 4, 6))
        with assert_raises(TypeError):
            f2 = ift.Field(None, None)
        with assert_raises(ValueError):
            f2 = ift.Field(s1, None)
        with assert_raises(ValueError):
            f1.imag
        with assert_raises(TypeError):
            f1.vdot(42)
        with assert_raises(ValueError):
            f1.vdot(ift.Field(s2, 1.))
        with assert_raises(TypeError):
            f1.copy_content_from(1)
        with assert_raises(ValueError):
            f1.copy_content_from(ift.Field(s2, 1.))
        with assert_raises(TypeError):
            ift.full(s1, [2, 3])

    def test_stdfunc(self):
        s = ift.RGSpace((200,))
        f = ift.Field(s, 27)
        assert_equal(f.local_data, 27)
        assert_equal(f.shape, (200,))
        assert_equal(f.dtype, np.int)
        fx = ift.empty(f.domain, f.dtype)
        assert_equal(f.dtype, fx.dtype)
        assert_equal(f.shape, fx.shape)
        fx = ift.full(f.domain, 0)
        assert_equal(f.dtype, fx.dtype)
        assert_equal(f.shape, fx.shape)
        assert_equal(fx.local_data, 0)
        fx = ift.full(f.domain, 1)
        assert_equal(f.dtype, fx.dtype)
        assert_equal(f.shape, fx.shape)
        assert_equal(fx.local_data, 1)
        fx = ift.full(f.domain, 67.)
        assert_equal(f.shape, fx.shape)
        assert_equal(fx.local_data, 67.)
        f = ift.Field.from_random("normal", s)
        f2 = ift.Field.from_random("normal", s)
        assert_equal((f > f2).local_data, f.local_data > f2.local_data)
        assert_equal((f >= f2).local_data, f.local_data >= f2.local_data)
        assert_equal((f < f2).local_data, f.local_data < f2.local_data)
        assert_equal((f <= f2).local_data, f.local_data <= f2.local_data)
        assert_equal((f != f2).local_data, f.local_data != f2.local_data)
        assert_equal((f == f2).local_data, f.local_data == f2.local_data)
        assert_equal((f+f2).local_data, f.local_data+f2.local_data)
        assert_equal((f-f2).local_data, f.local_data-f2.local_data)
        assert_equal((f*f2).local_data, f.local_data*f2.local_data)
        assert_equal((f/f2).local_data, f.local_data/f2.local_data)
        assert_equal((-f).local_data, -(f.local_data))
        assert_equal(abs(f).local_data, abs(f.local_data))
