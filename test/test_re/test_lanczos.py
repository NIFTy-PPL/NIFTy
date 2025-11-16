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
    tridiag, vecs = jft.lanczos.lanczos_tridiag(partial(matmul, m), v, shape0)
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
