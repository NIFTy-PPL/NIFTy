#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import math

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest
from nifty.re.ncx2_likelihood import FisherTable, _fisher_quad, fisher_asymptotic
from nifty.re.num.ncx2_jax import (
    _make_ncx2_logpdf_autodiff,
    log_itilde,
    make_ncx2_logpdf,
)
from numpy.testing import assert_allclose
from scipy import stats

jax.config.update("jax_enable_x64", True)

pmp = pytest.mark.parametrize

# small tables keep the test suite fast; accuracy of the default tables is not
# what is tested here
N_TAB = 256

# every Bessel branch: closed forms (k <= 4), long series + Hankel (5 <= k <= 21)
# and Debye (k >= 22)
K_ALL = (1, 2, 3, 4, 5, 8, 21, 22, 50)
K_SCORE = (5, 8, 21, 22, 50)


def _scaled_err(ours, ref):
    return np.abs(ours - ref) / np.maximum(1.0, np.abs(ref))


@pmp("k", K_ALL)
def test_logpdf_vs_scipy(k):
    logpdf = make_ncx2_logpdf(k)
    for lam in (0.0, 1e-3, 0.5, 3.0, 40.0, 1e3, 1e5):
        s = np.sqrt(lam + k) + np.linspace(-6.0, 10.0, 33)
        d = s[s > 0.05] ** 2
        ours = np.asarray(logpdf(d, lam))
        if lam == 0.0:
            ref = stats.chi2.logpdf(d, k)
        else:
            ref = stats.ncx2.logpdf(d, k, lam)
        ok = np.isfinite(ref)
        assert np.all(np.isfinite(ours))
        assert np.max(_scaled_err(ours[ok], ref[ok]), initial=0.0) < 1e-12


@pmp("nu", (-0.5, 0.0, 0.5, 1.0, 1.5, 4.0, 9.5, 10.0, 24.0))
def test_log_itilde_at_zero(nu):
    assert_allclose(log_itilde(jnp.array(0.0), nu), -math.lgamma(nu + 1.0))


@pmp("k", (1, 2, 3, 5, 22))
def test_logpdf_edges(k):
    logpdf = make_ncx2_logpdf(k)
    lam = jnp.array([0.0, 2.0])
    at_zero = np.asarray(logpdf(0.0, lam))
    if k == 1:
        assert np.all(np.isposinf(at_zero))
    elif k == 2:
        assert_allclose(at_zero, np.log(0.5) - np.asarray(lam) / 2.0)
    else:
        assert np.all(np.isneginf(at_zero))
    assert np.isneginf(logpdf(-1.0, 2.0))
    assert np.isnan(logpdf(1.0, -1.0))


@pmp("k", K_SCORE)
def test_score_vs_autodiff(k):
    rng = np.random.default_rng(42)
    lam = np.exp(rng.uniform(-6.0, 12.0, size=200))
    d = stats.ncx2.rvs(k, lam, random_state=rng)
    fast = jax.vmap(jax.grad(make_ncx2_logpdf(k), argnums=1))
    ref = jax.vmap(jax.grad(_make_ncx2_logpdf_autodiff(k), argnums=1))
    assert_allclose(fast(d, lam), ref(d, lam), rtol=1e-9, atol=1e-11)


@pmp("k", (1, 2, 3, 5, 22))
def test_score_finite_at_lam_zero(k):
    d = jnp.array([0.1, 1.0, 5.0, 30.0])
    g = jax.vmap(jax.grad(make_ncx2_logpdf(k), argnums=1))(d, jnp.zeros_like(d))
    assert np.all(np.isfinite(g))


@pmp("k", (1, 2, 5, 22))
def test_fisher_anchors(k):
    assert_allclose(_fisher_quad(k, [0.0]), 1.0 / (2.0 * k), rtol=1e-10)
    assert_allclose(_fisher_quad(k, [1e4]), fisher_asymptotic(k, 1e4), rtol=1e-6)


def _lh(k, n=N_TAB, lam_max=None, size=8, seed=0):
    rng = np.random.default_rng(seed)
    sigma2 = jnp.full(size, 0.5)
    lam = np.exp(rng.uniform(-3.0, 8.0, size=size))
    data = jnp.asarray(stats.ncx2.rvs(k, lam, random_state=rng)) * sigma2
    return jft.NoncentralChiSquared(data, sigma2, k, n=n, lam_max=lam_max)


@pmp("k", (1, 2, 5))
def test_transformation_derivative_is_sqrt_fisher(k):
    lh = _lh(k, lam_max=1e3)
    lam = jnp.array([0.0, 1e-6, 0.3, 7.0, 999.0, 1e3 * (1 + 1e-9), 5e4, 1e7])
    dg = jax.vmap(jax.grad(lh._transformation_lam))(lam)
    assert_allclose(dg, lh._sqrt_fisher_lam(lam), rtol=1e-12)
    # continuity of the transformation at lam_max
    eps = 1e-9
    lo = lh._transformation_lam(lh.lam_max * (1 - eps))
    hi = lh._transformation_lam(lh.lam_max * (1 + eps))
    assert_allclose(hi, lo, rtol=1e-8)


@pmp("k", (2, 5))
def test_energy_matches_logpdf_and_jit(k):
    lh = _lh(k)
    s = jnp.linspace(0.0, 50.0, 8)
    ref = -jnp.sum(make_ncx2_logpdf(k)(lh.d, s / lh.sigma2))
    assert_allclose(lh.energy(s), ref)
    assert_allclose(jax.jit(lambda lk, x: lk.energy(x))(lh, s), ref)


def test_pytree_data():
    rng = np.random.default_rng(1)
    data = {"a": jnp.asarray(rng.exponential(size=3)), "b": jnp.ones((2, 2))}
    sigma2 = {"a": jnp.full(3, 0.5), "b": jnp.full((2, 2), 2.0)}
    lh = jft.NoncentralChiSquared(data, sigma2, 3, n=N_TAB)
    s = {"a": jnp.full(3, 1.5), "b": jnp.full((2, 2), 4.0)}
    assert np.isfinite(lh.energy(s))
    t = jax.tree_util.tree_map(jnp.ones_like, s)
    jax.tree_util.tree_map(
        lambda a, b: assert_allclose(a, b**2),
        lh.metric(s, t),
        lh.left_sqrt_metric(s, t),
    )


def test_input_validation():
    one = jnp.ones(3)
    with pytest.raises(ValueError):
        jft.NoncentralChiSquared(-one, one, 2, n=N_TAB)
    with pytest.raises(ValueError):
        jft.NoncentralChiSquared(jnp.zeros(3), one, 3, n=N_TAB)
    with pytest.raises(ValueError):
        jft.NoncentralChiSquared(one, jnp.zeros(3), 2, n=N_TAB)
    # d = 0 is allowed for k = 2, where the density is finite
    jft.NoncentralChiSquared(jnp.zeros(3), one, 2, n=N_TAB)
    for k in (0, 1.5, True):
        with pytest.raises(ValueError):
            make_ncx2_logpdf(k)
    table = FisherTable.load_or_build(3, n=N_TAB)
    with pytest.raises(ValueError):
        jft.NoncentralChiSquared(one, one, 2, table=table)
    with pytest.raises(ValueError):
        jft.NoncentralChiSquared(one, one, 3, table=table, n=2 * N_TAB)


def test_table_save_load(tmp_path):
    table = FisherTable.load_or_build(2, n=N_TAB)
    path = tmp_path / "fisher_k2.npz"
    table.save(path)
    loaded = FisherTable.load(path)
    assert loaded.k == table.k and loaded.h == table.h
    assert_allclose(loaded.R, table.R)
    assert_allclose(loaded.G, table.G)
    lh = jft.NoncentralChiSquared(jnp.ones(3), jnp.ones(3), 2, table_path=path)
    assert lh._n_tab == N_TAB
