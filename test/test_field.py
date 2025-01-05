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
# Copyright(C) 2024-2025 Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises

from .common import setup_function, teardown_function, list2fixture

pmp = pytest.mark.parametrize
SPACES = [ift.RGSpace((4,)), ift.RGSpace((5))]
SPACE_COMBINATIONS = [(), SPACES[0], SPACES[1], SPACES]

device_id = list2fixture([-1, 0] if ift.device_available() else [-1])


@pmp('domain', SPACE_COMBINATIONS)
@pmp('attribute_desired_type',
     [['domain', ift.DomainTuple], ['val', ift.AnyArray],
      ['shape', tuple], ['size', (int, np.int64)]])
def test_return_types(domain, attribute_desired_type, device_id):
    attribute = attribute_desired_type[0]
    desired_type = attribute_desired_type[1]
    f = ift.Field.full(domain, 1., device_id)
    assert isinstance(getattr(f, attribute), desired_type)


@pmp('domain', SPACE_COMBINATIONS)
@pmp('attribute_desired_type', [['val_rw', ift.AnyArray],
                                ['asnumpy', np.ndarray],
                                ['asnumpy_rw', np.ndarray]])
def test_return_types2(domain, attribute_desired_type, device_id):
    attribute = attribute_desired_type[0]
    desired_type = attribute_desired_type[1]
    f = ift.Field.full(domain, 1., device_id)
    assert isinstance(getattr(f, attribute)(), desired_type)


def test_writeable(device_id):
    domain = ift.RGSpace(10)
    anyarr = ift.Field.full(domain, 1., device_id).val
    with assert_raises(ValueError):
        anyarr[0] = 12
    numpyarr = anyarr.asnumpy()
    with assert_raises(ValueError):
        numpyarr[0] = 12
    devicearr = anyarr.val
    if device_id > -1:
        # Cupy does not properly support readonly yet. See https://github.com/cupy/cupy/issues/2616
        return
    with assert_raises(ValueError):
        devicearr[0] = 12


def _spec1(k):
    return 42/(1. + k)**2


def _spec2(k):
    return 42/(1. + k)**3


@pmp('space1', [ift.RGSpace((8,), harmonic=True),
                ift.RGSpace((8, 8), harmonic=True, distances=0.123)])
@pmp('space2', [ift.RGSpace((8,), harmonic=True), ift.LMSpace(12)])
def test_power_synthesize_analyze(space1, space2, device_id):
    p1 = ift.PowerSpace(space1)
    fp1 = ift.PS_field(p1, _spec1, device_id)
    p2 = ift.PowerSpace(space2)
    fp2 = ift.PS_field(p2, _spec2, device_id)
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
    assert_allclose(sc1.mean.asnumpy(), fp1.asnumpy(), rtol=0.2)
    assert_allclose(sc2.mean.asnumpy(), fp2.asnumpy(), rtol=0.2)


@pmp('space1', [
    ift.RGSpace((8,), harmonic=True),
    ift.RGSpace((8, 8), harmonic=True, distances=0.123)
])
@pmp('space2', [ift.RGSpace((8,), harmonic=True), ift.LMSpace(12)])
def test_DiagonalOperator_power_analyze2(space1, space2, device_id):
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

    assert_allclose(sc1.mean.asnumpy(), fp1.asnumpy(), rtol=0.2)
    assert_allclose(sc2.mean.asnumpy(), fp2.asnumpy(), rtol=0.2)


@pmp('space', [
    ift.RGSpace((8,), harmonic=True), (),
    ift.RGSpace((8, 8), harmonic=True, distances=0.123),
    ift.RGSpace((2, 3, 7))
])
def test_norm(space, device_id):
    f = ift.Field.from_random(domain=space, random_type="normal",
                              dtype=np.complex128, device_id=device_id)
    gd = f.asnumpy().reshape(-1)
    assert_allclose(f.norm(), np.linalg.norm(gd))
    assert_allclose(f.norm(1), np.linalg.norm(gd, ord=1))
    assert_allclose(f.norm(2), np.linalg.norm(gd, ord=2))
    assert_allclose(f.norm(3), np.linalg.norm(gd, ord=3))
    assert_allclose(f.norm(np.inf), np.linalg.norm(gd, ord=np.inf))


def test_vdot():
    s = ift.RGSpace((10,))
    f1 = ift.Field.from_random(domain=s, random_type="normal", dtype=np.complex128)
    f2 = ift.Field.from_random(domain=s, random_type="normal", dtype=np.complex128)
    assert_allclose(f1.s_vdot(f2), f1.vdot(f2, spaces=0).asnumpy())
    assert_allclose(f1.s_vdot(f2), np.conj(f2.s_vdot(f1)))


def test_vdot2():
    x1 = ift.RGSpace((200,))
    x2 = ift.RGSpace((150,))
    m = ift.Field.full((x1, x2), .5)
    res = m.vdot(m, spaces=1)
    assert_allclose(res.asnumpy(), 37.5)


def test_outer():
    x1 = ift.RGSpace((9,))
    x2 = ift.RGSpace((3,))
    m1 = ift.Field.full(x1, .5)
    m2 = ift.Field.full(x2, 3.)
    res = m1.outer(m2)
    assert_allclose(res.asnumpy(), np.full((9, 3), 1.5))


def test_sum():
    x1 = ift.RGSpace((9,), distances=2.)
    x2 = ift.RGSpace((2, 12), distances=(0.3,))
    m1 = ift.Field(ift.makeDomain(x1), np.arange(9))
    m2 = ift.Field.full(ift.makeDomain((x1, x2)), 0.45)
    res1 = m1.s_sum()
    res2 = m2.sum(spaces=1)
    assert_allclose(res1, 36)
    assert_allclose(res2.asnumpy(), np.full(9, 2*12*0.45))


def test_integrate():
    x1 = ift.RGSpace((9,), distances=2.)
    x2 = ift.RGSpace((2, 12), distances=(0.3,))
    m1 = ift.Field(ift.makeDomain(x1), np.arange(9))
    m2 = ift.Field.full(ift.makeDomain((x1, x2)), 0.45)
    res1 = m1.s_integrate()
    res2 = m2.integrate(spaces=1)
    assert_allclose(res1, 36*2)
    assert_allclose(res2.asnumpy(), np.full(9, 2*12*0.45*0.3**2))
    for m in [m1, m2]:
        res3 = m.integrate()
        res4 = m.s_integrate()
        assert_allclose(res3.asnumpy(), res4)
    dom = ift.HPSpace(3)
    assert_allclose(ift.full(dom, 1).s_integrate(), 4*np.pi)


def test_dataconv():
    s1 = ift.RGSpace((10,))
    gd = np.arange(s1.shape[0])
    assert_equal(gd, ift.makeField(s1, gd).asnumpy())


def test_cast_domain():
    s1 = ift.RGSpace((10,))
    s2 = ift.RGSpace((10,), distances=20.)
    d = np.arange(s1.shape[0])
    d2 = ift.makeField(s1, d).cast_domain(s2).asnumpy()
    assert_equal(d, d2)


def test_empty_domain():
    f = ift.Field.full((), 5)
    assert_equal(f.asnumpy(), 5)
    f = ift.Field.full(None, 5)
    assert_equal(f.asnumpy(), 5)


def test_trivialities():
    s1 = ift.RGSpace((10,))
    f1 = ift.Field.full(s1, 27)
    assert_equal(f1.clip(a_min=29, a_max=50).asnumpy(), 29.)
    assert_equal(f1.clip(a_min=0, a_max=25).asnumpy(), 25.)
    assert_equal(f1.asnumpy(), f1.real.asnumpy())
    assert_equal(f1.asnumpy(), (+f1).asnumpy())
    f1 = ift.Field.full(s1, 27. + 3j)
    assert_equal(f1.ptw("reciprocal").asnumpy(), (1./f1).asnumpy())
    assert_equal(f1.real.asnumpy(), 27.)
    assert_equal(f1.imag.asnumpy(), 3.)
    assert_equal(f1.s_sum(), f1.sum(0).asnumpy())
    assert_equal(f1.conjugate().asnumpy(),
                 ift.Field.full(s1, 27. - 3j).asnumpy())
    f1 = ift.makeField(s1, np.arange(10))
    # assert_equal(f1.min(), 0)
    # assert_equal(f1.max(), 9)
    assert_equal(f1.s_prod(), 0)


def test_weight():
    s1 = ift.RGSpace((10,))
    f = ift.Field.full(s1, 10.)
    f2 = f.weight(1)
    assert_equal(f.weight(1).asnumpy(), f2.asnumpy())
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
    assert_allclose(s1.mean(0).asnumpy(), 1.)
    assert_allclose(s1.s_var(), 0., atol=1e-14)
    assert_allclose(s1.var(0).asnumpy(), 0., atol=1e-14)
    assert_allclose(s1.s_std(), 0., atol=1e-14)
    assert_allclose(s1.std(0).asnumpy(), 0., atol=1e-14)


def test_err():
    s1 = ift.RGSpace((10,))
    s2 = ift.RGSpace((11,))
    f1 = ift.Field.full(s1, 27)
    with assert_raises(ValueError):
        f2 = ift.Field(ift.DomainTuple.make(s2), f1.asnumpy())
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


def test_stdfunc(device_id):
    s = ift.RGSpace((200,))
    f = ift.Field.full(s, 27, device_id)
    assert_equal(f.asnumpy(), 27)
    assert_equal(f.shape, (200,))
    assert_equal(f.dtype, np.int64)
    fx = ift.full(f.domain, 0, device_id)
    assert_equal(f.dtype, fx.dtype)
    assert_equal(f.shape, fx.shape)
    assert_equal(fx.asnumpy(), 0)
    fx = ift.full(f.domain, 1, device_id)
    assert_equal(f.dtype, fx.dtype)
    assert_equal(f.shape, fx.shape)
    assert_equal(fx.asnumpy(), 1)
    fx = ift.full(f.domain, 67., device_id)
    assert_equal(f.shape, fx.shape)
    assert_equal(fx.asnumpy(), 67.)
    f = ift.Field.from_random(s, "normal", device_id=device_id)
    f2 = ift.Field.from_random(s, "normal", device_id=device_id)
    assert_equal((f > f2).asnumpy(), f.asnumpy() > f2.asnumpy())
    assert_equal((f >= f2).asnumpy(), f.asnumpy() >= f2.asnumpy())
    assert_equal((f < f2).asnumpy(), f.asnumpy() < f2.asnumpy())
    assert_equal((f <= f2).asnumpy(), f.asnumpy() <= f2.asnumpy())
    assert_equal((f != f2).asnumpy(), f.asnumpy() != f2.asnumpy())
    assert_equal((f == f2).asnumpy(), f.asnumpy() == f2.asnumpy())
    assert_equal((f + f2).asnumpy(), f.asnumpy() + f2.asnumpy())
    assert_equal((f - f2).asnumpy(), f.asnumpy() - f2.asnumpy())
    assert_equal((f*f2).asnumpy(), f.asnumpy()*f2.asnumpy())
    assert_equal((f/f2).asnumpy(), f.asnumpy()/f2.asnumpy())
    assert_equal((-f).asnumpy(), -(f.asnumpy()))
    assert_equal(abs(f).asnumpy(), abs(f.asnumpy()))


def test_emptydomain(device_id):
    f = ift.Field.full((), 3., device_id)
    assert_equal(f.s_sum(), 3.)
    assert_equal(f.s_prod(), 3.)
    assert_equal(f.asnumpy(), 3.)
    assert_equal(f.asnumpy().shape, ())
    assert_equal(f.asnumpy().size, 1)
    assert_equal(f.s_vdot(f), 9.)


@pmp('num', [5.])
@pmp('dom', [ift.RGSpace((8,), harmonic=True), ()])
@pmp('func', [
    "exp", "log", "sin", "cos", "tan", "sinh", "cosh", "sinc", "absolute",
    "sign", "log10", "log1p", "expm1"
])
def test_funcs(num, dom, func, device_id):
    f = ift.Field.full(dom, num, device_id)
    res = f.ptw(func)
    res2 = getattr(np, func)(np.full(f.shape, num))
    if func == "sinc" and device_id > -1:
        # On cupy sinc(5) is ~4e-17... and not 0.
        assert_allclose(res.asnumpy(), res2, atol=1e-16)
    else:
        assert_allclose(res.asnumpy(), res2)


@pmp('rtype', ['normal', 'pm1', 'uniform'])
@pmp('dtype', [np.float64, np.complex128])
def test_from_random(rtype, dtype, device_id):
    sp = ift.RGSpace(3)
    ift.Field.from_random(sp, rtype, dtype=dtype, device_id=device_id)


def test_field_of_objects():
    arr = np.array(['x', 'y', 'z'])
    sp = ift.RGSpace(3)
    with assert_raises(TypeError):
        ift.Field(sp, arr)
