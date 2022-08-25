# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
from typing import Callable, Optional, Union

import jax
from jax import numpy as jnp
from jax import random

from .forest_util import ShapeWithDtype


def lanczos_tridiag(
    mat: Callable, shape_dtype_struct: ShapeWithDtype, order: int,
    key: jnp.ndarray
):
    """Compute the Lanczos decomposition into a tri-diagonal matrix and its
    corresponding orthonormal projection matrix.
    """
    tridiag = jnp.zeros((order, order), dtype=shape_dtype_struct.dtype)
    vecs = jnp.zeros(
        (order, ) + shape_dtype_struct.shape, dtype=shape_dtype_struct.dtype
    )

    v = random.normal(key, shape=shape_dtype_struct.shape)
    v = v / jnp.linalg.norm(v)
    vecs = vecs.at[0].set(v)

    # Zeroth iteration
    w = mat(v)
    if w.shape != shape_dtype_struct.shape:
        ve = f"shape of `mat(v)` {w.shape!r} incompatible with {shape_dtype_struct}"
        raise ValueError(ve)
    alpha = jnp.dot(w, v)
    tridiag = tridiag.at[(0, 0)].set(alpha)
    w -= alpha * v
    beta = jnp.linalg.norm(w)

    tridiag = tridiag.at[(0, 1)].set(beta)
    tridiag = tridiag.at[(1, 0)].set(beta)
    vecs = vecs.at[1].set(w / beta)

    def reortho_step(j, state):
        vecs, w = state

        tau = vecs[j, :].reshape(shape_dtype_struct.shape)
        coeff = jnp.dot(w, tau)
        w -= coeff * tau
        return vecs, w

    def lanczos_step(i, state):
        tridiag, vecs, beta = state

        v = vecs[i, :].reshape(shape_dtype_struct.shape)
        v_old = vecs[i - 1, :].reshape(shape_dtype_struct.shape)

        w = mat(v) - beta * v_old
        alpha = jnp.dot(w, v)
        tridiag = tridiag.at[(i, i)].set(alpha)
        w -= alpha * v

        # Full reorthogonalization
        vecs, w = jax.lax.fori_loop(0, i, reortho_step, (vecs, w))

        # TODO: Raise if lanczos vectors are independent i.e. `beta` small?
        beta = jnp.linalg.norm(w)

        tridiag = tridiag.at[(i, i + 1)].set(beta)
        tridiag = tridiag.at[(i + 1, i)].set(beta)
        vecs = vecs.at[i + 1].set(w / beta)

        return tridiag, vecs, beta

    tridiag, vecs, beta = jax.lax.fori_loop(
        1, order - 1, lanczos_step, (tridiag, vecs, beta)
    )

    # Final tridiag value and reorthogonalization
    v = vecs[order - 1, :].reshape(shape_dtype_struct.shape)
    v_old = vecs[order - 2, :].reshape(shape_dtype_struct.shape)
    w = mat(v) - beta * v_old
    alpha = jnp.dot(w, v)
    tridiag = tridiag.at[(order - 1, order - 1)].set(alpha)
    w -= alpha * v
    vecs, w = jax.lax.fori_loop(0, order - 1, reortho_step, (vecs, w))

    return (tridiag, vecs)


def stochastic_logdet_from_lanczos(
    tridiag_stack: jnp.ndarray, matrix_shape0: int, func: Callable = jnp.log
):
    """Computes a stochastic estimate of the log-determinate of a matrix using
    its Lanczos decomposition.

    Implemented via the stoachstic Lanczos quadrature.
    """
    eig_vals, eig_vecs = jnp.linalg.eigh(tridiag_stack)
    # TODO: Mask Eigenvalues <= 0?

    num_random_probes = tridiag_stack.shape[0]

    eig_ves_first_component = eig_vecs[..., 0, :]
    func_of_eig_vals = func(eig_vals)

    dot_products = jnp.sum(eig_ves_first_component**2 * func_of_eig_vals)
    return matrix_shape0 / float(num_random_probes) * dot_products


def stochastic_lq_logdet(
    mat: Union[jnp.ndarray, Callable],
    order: int,
    n_samples: int,
    key: Union[int, jnp.ndarray],
    *,
    shape0: Optional[int] = None,
    dtype=None
):
    """Computes a stochastic estimate of the log-determinate of a matrix using
    the stochastic Lanczos quadrature algorithm.
    """
    shape0 = shape0 if shape0 is not None else mat.shape[0]
    mat = mat.__matmul__ if not hasattr(mat, "__call__") else mat
    if not isinstance(key, jnp.ndarray):
        key = random.PRNGKey(key)
    keys = random.split(key, n_samples)

    lanczos = partial(lanczos_tridiag, mat, ShapeWithDtype(shape0, dtype))
    tridiags, _ = jax.vmap(lanczos, in_axes=(None, 0),
                           out_axes=(0, 0))(order, keys)
    return stochastic_logdet_from_lanczos(tridiags, shape0)
