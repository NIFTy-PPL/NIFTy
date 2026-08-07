#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import sys
from functools import partial
from operator import matmul

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpy.testing import assert_allclose
from scipy.spatial import distance_matrix

import nifty.re as jft
from nifty.re.num.lanczos import _slq_gauss_radau

jax.config.update("jax_enable_x64", True)

pmp = pytest.mark.parametrize


def matern_kernel(distance, scale, cutoff, dof):
    from jax.scipy.special import gammaln
    from scipy.special import kv

    reg_dist = jnp.sqrt(2 * dof) * distance / cutoff
    cov = (
        scale**2
        * 2 ** (1 - dof)
        / jnp.exp(gammaln(dof))
        * (reg_dist) ** dof
        * kv(dof, reg_dist)
    )
    # NOTE, this is not safe for differentiating because `cov` still may
    # contain NaNs
    return jnp.where(distance < 1e-8 * cutoff, scale**2, cov)


@pmp("seed", tuple(range(12, 44, 5)))
@pmp("shape0", (128, 64))
def test_lanczos_tridiag(seed, shape0):
    rng = np.random.default_rng(seed)
    rng_key = random.PRNGKey(rng.integers(12, 42))

    m = rng.normal(size=(shape0,) * 2)
    m = m @ m.T  # ensure positive-definiteness

    v = random.rademacher(rng_key, (shape0,), float)
    tridiag, vecs = jft.lanczos.lanczos_tridiag(partial(matmul, m), v, order=shape0)
    m_est = vecs.T @ tridiag @ vecs

    assert_allclose(m_est, m, atol=1e-13, rtol=1e-13)


def _random_pd_matrix(n, min_eigenvalue=0.1, *, seed=None):
    """
    Generates a random n x n positive definite matrix with stable slogdet.
    Ensures minimum eigenvalue is at least min_eigenvalue for stability.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n))
    tril = np.tril(x)
    diag = np.diag(tril).clip(min_eigenvalue, None)
    np.fill_diagonal(tril, diag)
    sym = tril @ tril.T
    return sym


@pmp("seed", tuple(range(12, 44, 5)))
@pmp("shape0", (128, 64))
def test_stochastic_lq_logdet(seed, shape0, lq_order=15, n_lq_samples=10):
    rng = np.random.default_rng(seed)
    rng_key = random.PRNGKey(rng.integers(12, 42))
    m = _random_pd_matrix(shape0, min_eigenvalue=1.0, seed=rng.integers(0, 10))
    m *= 10  # make eigenvalues a bit larger

    _, logdet = jnp.linalg.slogdet(m)
    logdet_est = jft.stochastic_lq_logdet(m, lq_order, n_lq_samples, rng_key)
    assert_allclose(logdet_est, logdet, rtol=0.8, atol=10.0)
    print(f"{logdet=} :: {logdet_est=}", file=sys.stderr)


@pmp("as_callable", [False, True])
def test_slq_exact_diagonal_with_extra_function_and_remainder_batch(as_callable):
    diagonal = jnp.array([1.25, 2.0, 3.5, 6.0])
    matrix = jnp.diag(diagonal)
    operator = (lambda x: matrix @ x) if as_callable else matrix
    kwargs = {"n": diagonal.size} if as_callable else {}

    result = _slq_gauss_radau(
        operator,
        jnp.log,
        order=diagonal.size,
        num_samples=5,
        key=random.PRNGKey(0),
        probe_batch_size=2,
        extra_fns={"inv": lambda x: 1.0 / x - 1.0},
        **kwargs,
    )

    assert_allclose(result["estimate"], jnp.sum(jnp.log(diagonal)), atol=1e-12)
    assert_allclose(
        result["extra_inv_estimate"],
        jnp.sum(1.0 / diagonal - 1.0),
        atol=1e-12,
    )
    assert_allclose(result["stochastic_se"], 0.0, atol=1e-12)
    assert "radau_estimate" not in result
    assert "radau_lo" not in result


def test_slq_handles_early_breakdown():
    diagonal = jnp.full((4,), 2.5)
    result = _slq_gauss_radau(
        jnp.diag(diagonal),
        jnp.log,
        order=4,
        num_samples=3,
        key=random.PRNGKey(1),
    )
    assert_allclose(result["estimate"], 4 * jnp.log(2.5), atol=1e-12)


def test_slq_deflates_exact_eigenvector():
    diagonal = jnp.array([1.5, 2.5, 4.0, 7.0])
    deflate = jnp.eye(4)[:, -1:]
    result = _slq_gauss_radau(
        jnp.diag(diagonal),
        jnp.log,
        order=3,
        num_samples=4,
        key=random.PRNGKey(2),
        deflate_eigvecs=deflate,
    )
    assert_allclose(result["estimate"], jnp.sum(jnp.log(diagonal[:-1])), atol=1e-12)


def test_slq_jit_matches_eager():
    matrix = jnp.diag(jnp.array([1.25, 2.0, 3.5]))

    def run(key):
        return _slq_gauss_radau(
            matrix,
            jnp.log,
            order=3,
            num_samples=5,
            key=key,
            probe_batch_size=2,
        )

    key = random.PRNGKey(3)
    eager = run(key)
    compiled = jax.jit(run)(key)
    for name in eager:
        assert_allclose(compiled[name], eager[name], atol=1e-12)


def test_slq_uses_float32_when_x64_is_disabled():
    x64_enabled = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
        diagonal = jnp.ones(4, dtype=jnp.float32)
        result = _slq_gauss_radau(
            jnp.diag(diagonal),
            jnp.log,
            order=4,
            num_samples=2,
            key=random.PRNGKey(4),
        )
        assert result["estimate"].dtype == jnp.float32
        assert jnp.isfinite(result["estimate"])
        assert_allclose(result["estimate"], jnp.sum(jnp.log(diagonal)), rtol=2e-6)
    finally:
        jax.config.update("jax_enable_x64", x64_enabled)


@pmp("reorthogonalize", ["partial", "full"])
def test_slq_reorthogonalization_modes_are_traceable(reorthogonalize):
    diagonal = jnp.array([1.25, 2.0, 3.5, 6.0])

    def run(key):
        return _slq_gauss_radau(
            jnp.diag(diagonal),
            jnp.log,
            order=diagonal.size,
            num_samples=3,
            key=key,
            reorthogonalize=reorthogonalize,
            reorth_k=2,
        )

    key = random.PRNGKey(5)
    eager = run(key)
    compiled = jax.jit(run)(key)
    expected = jnp.sum(jnp.log(diagonal))

    assert_allclose(eager["estimate"], expected, atol=1e-12)
    assert_allclose(compiled["estimate"], expected, atol=1e-12)


def test_slq_order_one_radau_uses_forced_endpoints():
    result = _slq_gauss_radau(
        jnp.diag(jnp.array([1.0, 4.0])),
        jnp.log,
        order=1,
        num_samples=3,
        key=random.PRNGKey(6),
        lam_min=1.0,
        lam_max=4.0,
        compute_radau=True,
    )

    assert_allclose(result["radau_lo"], 0.0, atol=1e-12)
    assert_allclose(result["radau_hi"], 2.0 * jnp.log(4.0), atol=1e-12)
    assert result["quadrature_width"] > 0.0


def test_slq_radau_is_opt_in_and_auto_endpoint_adds_one_lanczos_pass():
    matrix = jnp.diag(jnp.array([1.25, 2.0, 3.5]))

    def run(*, compute_radau):
        calls = []

        def count_matvec(x):
            jax.debug.callback(lambda _: calls.append(None), x)
            return matrix @ x

        result = _slq_gauss_radau(
            count_matvec,
            jnp.log,
            order=3,
            num_samples=3,
            key=random.PRNGKey(5),
            n=3,
            probe_batch_size=2,
            compute_radau=compute_radau,
        )
        result["estimate"].block_until_ready()
        return result, len(calls)

    gauss, gauss_calls = run(compute_radau=False)
    radau, radau_calls = run(compute_radau=True)

    assert "radau_estimate" not in gauss
    assert "radau_estimate" in radau
    assert radau_calls == 2 * gauss_calls
