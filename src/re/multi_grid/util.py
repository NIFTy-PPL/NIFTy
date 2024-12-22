#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import jax
from jax import numpy as jnp


def _check(v, cut=1e-16):
    return v > cut


@jax.custom_jvp
def solve(A, X):
    assert A.ndim == 2
    assert (X.ndim == 1) or (X.ndim == 2)
    v, U = jnp.linalg.eigh(A)
    vi = jax.lax.select(_check(v), 1.0 / v, jnp.zeros_like(v))
    res = U.T @ X
    res *= vi[:, jnp.newaxis] if X.ndim == 2 else vi
    return U @ res


@solve.defjvp
def solve_jvp(primals, tangents):
    # Note: Makes use of `stable_inverse` directly to enable stable higher order
    # derivatives. This is a tradeoff against saving compute for first order.
    (A, X), (dA, dX) = primals, tangents
    res = solve(A, X)
    return res, solve(A, dX - dA @ res)


def _get_sqrt(v, U):
    vsq = jax.lax.select(_check(v), jnp.sqrt(v), jnp.zeros_like(v))
    return U @ (vsq[:, jnp.newaxis] * U.T)


@jax.custom_jvp
def sqrtm(M):
    v, U = jnp.linalg.eigh(M)
    return _get_sqrt(v, U)


@sqrtm.defjvp
def sqrtm_jvp(M, dM):
    # Note: Only stable 1st derivative!
    M, dM = M[0], dM[0]
    v, U = jnp.linalg.eigh(M)

    dM = U.T @ dM @ U
    valid = _check(v)
    vsq = jnp.sqrt(jax.lax.select(valid, v, jnp.ones_like(v)))
    dres = jax.lax.select(
        valid[:, jnp.newaxis] & valid[jnp.newaxis, :],
        dM / (vsq[:, jnp.newaxis] + vsq[jnp.newaxis, :]),
        jnp.zeros_like(dM),
    )
    return _get_sqrt(v, U), U @ dres @ U.T
