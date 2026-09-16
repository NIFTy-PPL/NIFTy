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
# Copyright(C) 2026 Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

import nifty.cl as ift
from nifty.cl.extra import assert_allclose as ift_assert_allclose

from .common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize

_rgspace = ift.RGSpace(7)
_multidomain = ift.makeDomain({"a": ift.RGSpace(4),
                               "b": ift.UnstructuredDomain((2, 3))})

domain = list2fixture([_rgspace, ift.RGSpace((3, 4)), _multidomain])
dtype = list2fixture([np.float32, np.float64, np.complex64, np.complex128])
device_id = list2fixture([-1, 0] if ift.device_available() else [-1])

_rtol = {np.dtype(np.float32): 1e-4, np.dtype(np.complex64): 1e-4,
         np.dtype(np.float64): 1e-11, np.dtype(np.complex128): 1e-11}


def _samples(domain, n, dtype, device_id=-1):
    return [ift.from_random(domain, "normal", dtype=dtype, device_id=device_id)
            for _ in range(n)]


def _np_mean_var(samples):
    """Reference mean and unbiased variance, computed with numpy."""
    domain = samples[0].domain
    if isinstance(domain, ift.MultiDomain):
        mean = {kk: np.mean([ss.asnumpy()[kk] for ss in samples], axis=0)
                for kk in domain.keys()}
        var = {kk: np.var([ss.asnumpy()[kk] for ss in samples], axis=0, ddof=1)
               for kk in domain.keys()}
    else:
        np_samples = [ss.asnumpy() for ss in samples]
        mean = np.mean(np_samples, axis=0)
        var = np.var(np_samples, axis=0, ddof=1)
    return ift.makeField(domain, mean), ift.makeField(domain, var)


@pmp("nsamples", [2, 3, 17])
def test_mean_var(domain, dtype, nsamples):
    samples = _samples(domain, nsamples, dtype)
    sc = ift.StatCalculator()
    for ss in samples:
        sc.add(ss)
    mean, var = _np_mean_var(samples)
    rtol = _rtol[np.dtype(dtype)]
    ift_assert_allclose(sc.mean, mean, rtol)
    # The variance of complex samples is real-valued up to rounding errors.
    ift_assert_allclose(sc.var.real if np.issubdtype(dtype, np.complexfloating)
                     else sc.var, var, rtol)


def test_running_statistics(domain):
    """Mean and variance are correct after every single `add`."""
    samples = _samples(domain, 8, np.float64)
    sc = ift.StatCalculator()
    for ii, ss in enumerate(samples):
        sc.add(ss)
        assert_equal(sc.mean.domain, ss.domain)
        if ii == 0:
            ift_assert_allclose(sc.mean, ss, 1e-14)
            with pytest.raises(RuntimeError):
                sc.var
            continue
        mean, var = _np_mean_var(samples[:ii+1])
        ift_assert_allclose(sc.mean, mean, 1e-11)
        ift_assert_allclose(sc.var, var, 1e-11)


def test_no_samples():
    sc = ift.StatCalculator()
    with pytest.raises(RuntimeError):
        sc.mean
    with pytest.raises(RuntimeError):
        sc.var


def test_single_sample(domain):
    sc = ift.StatCalculator()
    fld = ift.from_random(domain)
    sc.add(fld)
    ift_assert_allclose(sc.mean, fld, 1e-14)
    with pytest.raises(RuntimeError):
        sc.var


def test_identical_samples(domain):
    """The variance of identical samples vanishes exactly."""
    fld = ift.from_random(domain)
    sc = ift.StatCalculator()
    for _ in range(5):
        sc.add(fld)
    ift_assert_allclose(sc.mean, fld, 1e-14)
    ift.extra.assert_equal(sc.var, ift.full(domain, 0))


def test_dtype_and_domain_preserved(domain, dtype):
    """Neither mean nor variance change precision or domain."""
    sc = ift.StatCalculator()
    for ss in _samples(domain, 4, dtype):
        sc.add(ss)
    fld = ift.from_random(domain, dtype=dtype)
    for res in [sc.mean, sc.var]:
        assert_equal(res.domain, fld.domain)
        assert_equal(res.dtype, fld.dtype)


def test_device(domain, device_id):
    sc = ift.StatCalculator()
    for ss in _samples(domain, 4, np.float64, device_id):
        sc.add(ss)
    for res in [sc.mean, sc.var]:
        assert_equal(res.device_id,
                     ift.from_random(domain, device_id=device_id).device_id)


def test_sc():
    """The variance of complex samples is defined via the complex conjugate."""
    sc = ift.StatCalculator()
    sc.add(ift.Field.scalar(1j))
    sc.add(ift.Field.scalar(-1j))
    np.testing.assert_equal(sc.var.asnumpy(), 2)
    np.testing.assert_equal(sc.mean.asnumpy(), 0)


@pmp("dtype", [np.complex64, np.complex128])
def test_complex_variance_is_real_and_positive(domain, dtype):
    sc = ift.StatCalculator()
    samples = _samples(domain, 6, dtype)
    for ss in samples:
        sc.add(ss)
    var = sc.var.asnumpy()
    for vv in (var.values() if isinstance(var, dict) else [var]):
        assert np.all(vv.real > 0)
        assert_allclose(vv.imag, 0., atol=1e-4*np.max(vv.real))


def test_complex_variance_value():
    """Purely imaginary deviations contribute positively to the variance."""
    dom = ift.UnstructuredDomain(3)
    samples = [ift.makeField(dom, np.array([1+1j, 2-3j, 1j])),
               ift.makeField(dom, np.array([1-1j, -2+3j, -1j])),
               ift.makeField(dom, np.array([3+0j, 0j, 0j]))]
    sc = ift.StatCalculator()
    for ss in samples:
        sc.add(ss)
    mean, var = _np_mean_var(samples)
    ift_assert_allclose(sc.mean, mean, 1e-13)
    ift_assert_allclose(sc.var.real, var, 1e-13)


def test_numerical_stability():
    """Welford's algorithm does not suffer from catastrophic cancellation.

    A naive one-pass estimator (`E[x^2] - E[x]^2`) would be off by O(1) for the
    values used here, the remaining error is the one of the input itself.
    """
    dom = ift.UnstructuredDomain(5)
    offset = 1e8
    rng = np.random.default_rng(42)
    arrs = [offset + rng.normal(size=dom.shape) for _ in range(100)]
    sc = ift.StatCalculator()
    for aa in arrs:
        sc.add(ift.makeField(dom, aa))
    assert_allclose(sc.mean.asnumpy(), np.mean(arrs, axis=0), rtol=1e-14)
    assert_allclose(sc.var.asnumpy(), np.var(arrs, axis=0, ddof=1), rtol=1e-7)


def test_unbiased_estimate():
    """Mean and variance converge to the ones of the sampled distribution."""
    sc = ift.StatCalculator()
    for _ in range(3000):
        sc.add(ift.from_random(_rgspace, "normal"))
    assert_allclose(np.mean(sc.mean.asnumpy()), 0., atol=0.05)
    assert_allclose(np.mean(sc.var.asnumpy()), 1., rtol=0.05)


def test_nonfield_types():
    """Anything that supports the necessary arithmetic can be added."""
    sc = ift.StatCalculator()
    for vv in [1., 2., 4.]:
        sc.add(vv)
    assert_allclose(sc.mean, 7/3)
    assert_allclose(sc.var, 7/3)

    sc = ift.StatCalculator()
    arrs = [np.array([1., 2.]), np.array([3., 4.]), np.array([5., 7.])]
    for aa in arrs:
        sc.add(aa)
    assert_allclose(sc.mean, np.mean(arrs, axis=0))
    assert_allclose(sc.var, np.var(arrs, axis=0, ddof=1))


def test_domain_mismatch(domain):
    sc = ift.StatCalculator()
    sc.add(ift.from_random(domain))
    with pytest.raises(ValueError):
        sc.add(ift.from_random(ift.RGSpace(11)))
