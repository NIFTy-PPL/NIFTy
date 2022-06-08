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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises

from .common import setup_function, teardown_function

pmp = pytest.mark.parametrize
SPACES = [ift.RGSpace((4,)), ift.RGSpace((5))]
SPACE_COMBINATIONS = [(), SPACES[0], SPACES[1], SPACES]


@pmp('domain', SPACE_COMBINATIONS)
@pmp('attribute_desired_type',
     [['domain', ift.DomainTuple], ['val', np.ndarray],
      ['shape', tuple], ['size', (int, np.int64)]])
def test_return_types(domain, attribute_desired_type):
    attribute = attribute_desired_type[0]
    desired_type = attribute_desired_type[1]
    f = ift.Field.full(domain, 1.)
    assert_equal(isinstance(getattr(f, attribute), desired_type), True)


def _spec1(k):
    return 42/(1. + k)**2


def _spec2(k):
    return 42/(1. + k)**3


@pmp('space1', [ift.RGSpace((8,), harmonic=True),
                ift.RGSpace((8, 8), harmonic=True, distances=0.123)])
@pmp('space2', [ift.RGSpace((8,), harmonic=True), ift.LMSpace(12)])
def test_power_synthesize_analyze(space1, space2):
    p1 = ift.PowerSpace(space1)
    fp1 = ift.PS_field(p1, _spec1)
    p2 = ift.PowerSpace(space2)
    fp2 = ift.PS_field(p2, _spec2)
    op1 = ift.create_power_operator((space1, space2), _spec1, 0, float)
    op2 = ift.create_power_operator((space1, space2), _spec2, 1, float)
    opfull = op2 @ op1

    samples = 120
    sc1 = ift.StatCalculator()
    sc2 = ift.StatCalculator()
    for ii in range(samples):
        sk = opfull.draw_sample()
        sp = ift.power_analyze(sk, spaces=(0, 1), keep_phase_information=False)
        sc1.add(sp.sum(spaces=1)/fp2.s_sum())
        sc2.add(sp.sum(spaces=0)/fp1.s_sum())
    assert_allclose(sc1.mean.val, fp1.val, rtol=0.2)
    assert_allclose(sc2.mean.val, fp2.val, rtol=0.2)


@pmp('space1', [
    ift.RGSpace((8,), harmonic=True),
    ift.RGSpace((8, 8), harmonic=True, distances=0.123)
])
@pmp('space2', [ift.RGSpace((8,), harmonic=True), ift.LMSpace(12)])
def test_DiagonalOperator_power_analyze2(space1, space2):
    fp1 = ift.PS_field(ift.PowerSpace(space1), _spec1)
    fp2 = ift.PS_field(ift.PowerSpace(space2), _spec2)

    S_1 = ift.create_power_operator((space1, space2), _spec1, 0, float)
    S_2 = ift.create_power_operator((space1, space2), _spec2, 1, float)
    S_full = S_2 @ S_1

    samples = 500
    sc1 = ift.StatCalculator()
    sc2 = ift.StatCalculator()

    for ii in range(samples):
        sk = S_full.draw_sample()
        sp = ift.power_analyze(sk, spaces=(0, 1), keep_phase_information=False)
        sc1.add(sp.sum(spaces=1)/fp2.s_sum())
        sc2.add(sp.sum(spaces=0)/fp1.s_sum())

    assert_allclose(sc1.mean.val, fp1.val, rtol=0.2)
    assert_allclose(sc2.mean.val, fp2.val, rtol=0.2)


@pmp('space', [
    ift.RGSpace((8,), harmonic=True), (),
    ift.RGSpace((8, 8), harmonic=True, distances=0.123),
    ift.RGSpace((2, 3, 7))
])
def test_norm(space):
    f = ift.Field.from_random(domain=space, random_type="normal", dtype=np.complex128)
    gd = f.val.reshape(-1)
    assert_allclose(f.norm(), np.linalg.norm(gd))
    assert_allclose(f.norm(1), np.linalg.norm(gd, ord=1))
    assert_allclose(f.norm(2), np.linalg.norm(gd, ord=2))
    assert_allclose(f.norm(3), np.linalg.norm(gd, ord=3))
    assert_allclose(f.norm(np.inf), np.linalg.norm(gd, ord=np.inf))


def test_vdot():
    s = ift.RGSpace((10,))
    f1 = ift.Field.from_random(domain=s, random_type="normal", dtype=np.complex128)
    f2 = ift.Field.from_random(domain=s, random_type="normal", dtype=np.complex128)
    assert_allclose(f1.s_vdot(f2), f1.vdot(f2, spaces=0).val)
    assert_allclose(f1.s_vdot(f2), np.conj(f2.s_vdot(f1)))


def test_vdot2():
    x1 = ift.RGSpace((200,))
    x2 = ift.RGSpace((150,))
    m = ift.Field.full((x1, x2), .5)
    res = m.vdot(m, spaces=1)
    assert_allclose(res.val, 37.5)


def test_outer():
    x1 = ift.RGSpace((9,))
    x2 = ift.RGSpace((3,))
    m1 = ift.Field.full(x1, .5)
    m2 = ift.Field.full(x2, 3.)
    res = m1.outer(m2)
    assert_allclose(res.val, np.full((9, 3), 1.5))


def test_sum():
    x1 = ift.RGSpace((9,), distances=2.)
    x2 = ift.RGSpace((2, 12), distances=(0.3,))
    m1 = ift.Field(ift.makeDomain(x1), np.arange(9))
    m2 = ift.Field.full(ift.makeDomain((x1, x2)), 0.45)
    res1 = m1.s_sum()
    res2 = m2.sum(spaces=1)
    assert_allclose(res1, 36)
    assert_allclose(res2.val, np.full(9, 2*12*0.45))


def test_integrate():
    x1 = ift.RGSpace((9,), distances=2.)
    x2 = ift.RGSpace((2, 12), distances=(0.3,))
    m1 = ift.Field(ift.makeDomain(x1), np.arange(9))
    m2 = ift.Field.full(ift.makeDomain((x1, x2)), 0.45)
    res1 = m1.s_integrate()
    res2 = m2.integrate(spaces=1)
    assert_allclose(res1, 36*2)
    assert_allclose(res2.val, np.full(9, 2*12*0.45*0.3**2))
    for m in [m1, m2]:
        res3 = m.integrate()
        res4 = m.s_integrate()
        assert_allclose(res3.val, res4)
    dom = ift.HPSpace(3)
    assert_allclose(ift.full(dom, 1).s_integrate(), 4*np.pi)


def test_dataconv():
    s1 = ift.RGSpace((10,))
    gd = np.arange(s1.shape[0])
    assert_equal(gd, ift.makeField(s1, gd).val)


def test_cast_domain():
    s1 = ift.RGSpace((10,))
    s2 = ift.RGSpace((10,), distances=20.)
    d = np.arange(s1.shape[0])
    d2 = ift.makeField(s1, d).cast_domain(s2).val
    assert_equal(d, d2)


def test_empty_domain():
    f = ift.Field.full((), 5)
    assert_equal(f.val, 5)
    f = ift.Field.full(None, 5)
    assert_equal(f.val, 5)


def test_trivialities():
    s1 = ift.RGSpace((10,))
    f1 = ift.Field.full(s1, 27)
    assert_equal(f1.clip(a_min=29, a_max=50).val, 29.)
    assert_equal(f1.clip(a_min=0, a_max=25).val, 25.)
    assert_equal(f1.val, f1.real.val)
    assert_equal(f1.val, (+f1).val)
    f1 = ift.Field.full(s1, 27. + 3j)
    assert_equal(f1.ptw("reciprocal").val, (1./f1).val)
    assert_equal(f1.real.val, 27.)
    assert_equal(f1.imag.val, 3.)
    assert_equal(f1.s_sum(), f1.sum(0).val)
    assert_equal(f1.conjugate().val,
                 ift.Field.full(s1, 27. - 3j).val)
    f1 = ift.makeField(s1, np.arange(10))
    # assert_equal(f1.min(), 0)
    # assert_equal(f1.max(), 9)
    assert_equal(f1.s_prod(), 0)


def test_weight():
    s1 = ift.RGSpace((10,))
    f = ift.Field.full(s1, 10.)
    f2 = f.weight(1)
    assert_equal(f.weight(1).val, f2.val)
    assert_equal(f.domain.total_volume(), 1)
    assert_equal(f.domain.total_volume(0), 1)
    assert_equal(f.domain.total_volume((0,)), 1)
    assert_equal(f.total_volume(), 1)
    assert_equal(f.total_volume(0), 1)
    assert_equal(f.total_volume((0,)), 1)
    assert_equal(f.domain.scalar_weight(), 0.1)
    assert_equal(f.domain.scalar_weight(0), 0.1)
    assert_equal(f.domain.scalar_weight((0,)), 0.1)
    assert_equal(f.scalar_weight(), 0.1)
    assert_equal(f.scalar_weight(0), 0.1)
    assert_equal(f.scalar_weight((0,)), 0.1)
    s1 = ift.GLSpace(10)
    f = ift.Field.full(s1, 10.)
    assert_equal(f.domain.scalar_weight(), None)
    assert_equal(f.domain.scalar_weight(0), None)
    assert_equal(f.domain.scalar_weight((0,)), None)


@pmp('dom', [ift.RGSpace(10), ift.GLSpace(10)])
@pmp('dt', [np.float64, np.complex128])
def test_reduction(dom, dt):
    s1 = ift.Field.full(dom, dt(1.))
    assert_allclose(s1.s_mean(), 1.)
    assert_allclose(s1.mean(0).val, 1.)
    assert_allclose(s1.s_var(), 0., atol=1e-14)
    assert_allclose(s1.var(0).val, 0., atol=1e-14)
    assert_allclose(s1.s_std(), 0., atol=1e-14)
    assert_allclose(s1.std(0).val, 0., atol=1e-14)


def test_err():
    s1 = ift.RGSpace((10,))
    s2 = ift.RGSpace((11,))
    f1 = ift.Field.full(s1, 27)
    with assert_raises(ValueError):
        f2 = ift.Field(ift.DomainTuple.make(s2), f1.val)
    with assert_raises(TypeError):
        f2 = ift.Field.full(s2, "xyz")
    with assert_raises(TypeError):
        if f1:
            pass
    with assert_raises(TypeError):
        f1.full((2, 4, 6))
    with assert_raises(TypeError):
        f2 = ift.Field(None, None)
    with assert_raises(TypeError):
        f2 = ift.Field(s1, None)
    with assert_raises(ValueError):
        f1.imag
    with assert_raises(TypeError):
        f1.vdot(42)
    with assert_raises(ValueError):
        f1.vdot(ift.Field.full(s2, 1.))
    with assert_raises(TypeError):
        ift.full(s1, [2, 3])
    with assert_raises(TypeError):
        ift.Field(s2, [0, 1])
    with assert_raises(TypeError):
        f1.outer([0, 1])
    with assert_raises(ValueError):
        f1.extract(s2)
    with assert_raises(TypeError):
        f1 += f1
    f2 = ift.Field.full(s2, 27)
    with assert_raises(ValueError):
        f1 + f2


def test_stdfunc():
    s = ift.RGSpace((200,))
    f = ift.Field.full(s, 27)
    assert_equal(f.val, 27)
    assert_equal(f.shape, (200,))
    assert_equal(f.dtype, np.int64)
    fx = ift.full(f.domain, 0)
    assert_equal(f.dtype, fx.dtype)
    assert_equal(f.shape, fx.shape)
    assert_equal(fx.val, 0)
    fx = ift.full(f.domain, 1)
    assert_equal(f.dtype, fx.dtype)
    assert_equal(f.shape, fx.shape)
    assert_equal(fx.val, 1)
    fx = ift.full(f.domain, 67.)
    assert_equal(f.shape, fx.shape)
    assert_equal(fx.val, 67.)
    f = ift.Field.from_random(s, "normal")
    f2 = ift.Field.from_random(s, "normal")
    assert_equal((f > f2).val, f.val > f2.val)
    assert_equal((f >= f2).val, f.val >= f2.val)
    assert_equal((f < f2).val, f.val < f2.val)
    assert_equal((f <= f2).val, f.val <= f2.val)
    assert_equal((f != f2).val, f.val != f2.val)
    assert_equal((f == f2).val, f.val == f2.val)
    assert_equal((f + f2).val, f.val + f2.val)
    assert_equal((f - f2).val, f.val - f2.val)
    assert_equal((f*f2).val, f.val*f2.val)
    assert_equal((f/f2).val, f.val/f2.val)
    assert_equal((-f).val, -(f.val))
    assert_equal(abs(f).val, abs(f.val))


def test_emptydomain():
    f = ift.Field.full((), 3.)
    assert_equal(f.s_sum(), 3.)
    assert_equal(f.s_prod(), 3.)
    assert_equal(f.val, 3.)
    assert_equal(f.val.shape, ())
    assert_equal(f.val.size, 1)
    assert_equal(f.s_vdot(f), 9.)


@pmp('num', [float(5), 5.])
@pmp('dom', [ift.RGSpace((8,), harmonic=True), ()])
@pmp('func', [
    "exp", "log", "sin", "cos", "tan", "sinh", "cosh", "sinc", "absolute",
    "sign", "log10", "log1p", "expm1"
])
def test_funcs(num, dom, func):
    num = 5
    f = ift.Field.full(dom, num)
    res = f.ptw(func)
    res2 = getattr(np, func)(num)
    assert_allclose(res.val, res2)


@pmp('rtype', ['normal', 'pm1', 'uniform'])
@pmp('dtype', [np.float64, np.complex128])
def test_from_random(rtype, dtype):
    sp = ift.RGSpace(3)
    ift.Field.from_random(sp, rtype, dtype=dtype)


def test_field_of_objects():
    arr = np.array(['x', 'y', 'z'])
    sp = ift.RGSpace(3)
    with assert_raises(TypeError):
        ift.Field(sp, arr)
