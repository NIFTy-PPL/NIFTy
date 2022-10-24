#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import pytest

pytest.importorskip("jax")

from functools import partial
import sys
from jax import random
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial import distance_matrix

import nifty8.re as jft

pmp = pytest.mark.parametrize


def matern_kernel(distance, scale, cutoff, dof):
    from jax.scipy.special import gammaln
    from scipy.special import kv

    reg_dist = jnp.sqrt(2 * dof) * distance / cutoff
    cov = scale**2 * 2**(1 - dof) / jnp.exp(
        gammaln(dof)
    ) * (reg_dist)**dof * kv(dof, reg_dist)
    # NOTE, this is not safe for differentiating because `cov` still may
    # contain NaNs
    return jnp.where(distance < 1e-8 * cutoff, scale**2, cov)


from operator import matmul


@pmp("seed", tuple(range(12, 44, 5)))
@pmp("shape0", (128, 64))
def test_lanczos_tridiag(seed, shape0):
    rng = np.random.default_rng(seed)
    rng_key = random.PRNGKey(rng.integers(12, 42))

    m = rng.normal(size=(shape0, ) * 2)
    m = m @ m.T  # ensure positive-definiteness

    v = random.rademacher(rng_key, (shape0, ), float)
    tridiag, vecs = jft.lanczos.lanczos_tridiag(partial(matmul, m), v, shape0)
    m_est = vecs.T @ tridiag @ vecs

    assert_allclose(m_est, m, atol=1e-13, rtol=1e-13)


@pmp("seed", tuple(range(12, 44, 5)))
@pmp("shape0", (128, 64))
def test_stochastic_lq_logdet(seed, shape0, lq_order=15, n_lq_samples=10):
    rng = np.random.default_rng(seed)
    rng_key = random.PRNGKey(rng.integers(12, 42))

    c = np.exp(3 + rng.normal())
    s = np.exp(rng.normal())

    p = np.logspace(np.log(0.1 * c), np.log(1e+2 * c), num=shape0 - 1)
    p = np.concatenate(([0], p)).reshape(-1, 1)

    m = jnp.asarray(
        matern_kernel(distance_matrix(p, p), cutoff=c, scale=s, dof=2.5)
    )

    _, logdet = jnp.linalg.slogdet(m)
    logdet_est = jft.stochastic_lq_logdet(m, lq_order, n_lq_samples, rng_key)
    assert_allclose(logdet_est, logdet, rtol=2., atol=20.)
    print(f"{logdet=} :: {logdet_est=}", file=sys.stderr)
