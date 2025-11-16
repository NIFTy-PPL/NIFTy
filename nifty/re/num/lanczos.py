# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from operator import matmul
from typing import Callable, Optional, TypeVar, Union

import jax
from jax import numpy as jnp
from jax import random
from jax.tree_util import Partial

from ..lax import fori_loop
from ..tree_math import ShapeWithDtype

V = TypeVar("V")


def lanczos_tridiag(
    mat: Callable[[jnp.ndarray], jnp.ndarray],
    v: jnp.ndarray,
    order: int,
    tol: float = 1e-12,
):
    """Compute the Lanczos decomposition into a tri-diagonal matrix and its
    corresponding orthonormal projection matrix.

    The tridiagonal matrix is of shape (order x order) and the stack of vectors
    of shape (order, n).
    * mat(v) must return a vector with same shape as v.
    * This version avoids NaNs by guarding beta==0 (Lanczos breakdown).
    * It keeps fixed shapes (pads with zeros) rather than terminating early.
    """
    # The implementation is inspired by
    # * https://en.wikipedia.org/wiki/Lanczos_algorithm with re-orthogonalization https://en.wikipedia.org/wiki/Lanczos_algorithm#Numerical_stability
    # * https://github.com/cornellius-gp/linear_operator/blob/main/linear_operator/utils/lanczos.py

    swd = ShapeWithDtype.from_leave(v)
    shape, dtype = swd.shape, swd.dtype
    if order < 1:
        raise ValueError("order must be >= 1")
    # TODO
    # * use `tree_math.dot` and `tree_math.norm` in favor of plain `jnp.dot`
    # * remove all reshapes as they are unnecessary
    tridiag = jnp.zeros((order, order), dtype=dtype)
    vecs = jnp.zeros((order,) + shape, dtype=dtype)

    v = v / jnp.linalg.norm(v)
    vecs = vecs.at[0].set(v)
    # Special-case order == 1
    if order == 1:
        w = mat(v)
        if w.shape != shape:
            raise ValueError(f"shape of mat(v) {w.shape!r} incompatible with {swd}")
        alpha = jnp.dot(w, v)
        tridiag = tridiag.at[(0, 0)].set(alpha)
        return tridiag, vecs

    # Zeroth iteration
    w = mat(v)
    if w.shape != shape:
        raise ValueError(f"shape of `mat(v)` {w.shape!r} incompatible with {swd}")
    alpha = jnp.dot(w, v)
    tridiag = tridiag.at[(0, 0)].set(alpha)
    w -= alpha * v
    beta = jnp.linalg.norm(w)

    # Guard small beta: if beta <= tol, set next vector to zeros (avoid div-by-zero)
    next_vec = jnp.where(beta > tol, w / beta, jnp.zeros_like(w))

    tridiag = tridiag.at[(0, 1)].set(beta).at[(1, 0)].set(beta)
    vecs = vecs.at[1].set(next_vec)

    def reortho_step(j, state):
        vecs, w = state
        tau = vecs[j].reshape(shape)
        coeff = jnp.dot(w, tau)
        w -= coeff * tau
        return vecs, w

    def lanczos_step(i, state):
        tridiag, vecs, beta = state

        v = vecs[i].reshape(shape)
        v_old = vecs[i - 1].reshape(shape)

        w = mat(v) - beta * v_old
        alpha = jnp.dot(w, v)
        tridiag = tridiag.at[(i, i)].set(alpha)
        w -= alpha * v

        # Full reorthogonalization
        # NOTE, in theory the loop could terminate at `i` but this would make
        # JAX's default backwards pass not work
        vecs, w = fori_loop(0, order, reortho_step, (vecs, w))

        beta = jnp.linalg.norm(w)

        # avoid dividing by zero: if beta_local small -> set next vec to zeros
        new_vec = jnp.where(beta > tol, w / beta, jnp.zeros_like(w))
        tridiag = tridiag.at[(i, i + 1)].set(beta).at[(i + 1, i)].set(beta)
        vecs = vecs.at[i + 1].set(new_vec)

        return tridiag, vecs, beta

    if order > 2:
        # loop i = 1 .. order-2 inclusive
        tridiag, vecs, beta = fori_loop(
            1, order - 1, lanczos_step, (tridiag, vecs, beta)
        )
    else:
        # when order == 2, we skip the internal loop and use beta from zeroth iteration
        pass

    # Final diagonal entry (index order-1)
    v = vecs[order - 1].reshape(shape)
    v_old = vecs[order - 2].reshape(shape)
    w = mat(v) - beta * v_old
    alpha = jnp.dot(w, v)
    tridiag = tridiag.at[(order - 1, order - 1)].set(alpha)
    w -= alpha * v
    vecs, w = fori_loop(0, order - 1, reortho_step, (vecs, w))

    # no final division if beta tiny (already avoided during loop)
    return tridiag, vecs


def stochastic_logdet_from_lanczos(
    tridiag_stack: jnp.ndarray,
    matrix_shape0: int,
    func: Callable = jnp.log,
    *,
    tol=1e-14,
):
    """Computes a stochastic estimate of the log-determinate of a matrix using
    its Lanczos decomposition.

    Implemented via the stoachstic Lanczos quadrature.
    """
    eig_vals, eig_vecs = jnp.linalg.eigh(tridiag_stack)
    eig_vals = jnp.where(eig_vals < tol, jnp.nan, eig_vals)

    num_random_probes = tridiag_stack.shape[0]

    eig_ves_first_component = eig_vecs[..., 0, :]
    func_of_eig_vals = func(eig_vals)

    dot_products = jnp.nansum(eig_ves_first_component**2 * func_of_eig_vals)
    return matrix_shape0 / float(num_random_probes) * dot_products


def stochastic_lq_logdet(
    mat: Union[jnp.ndarray, Callable],
    order: int,
    n_samples: int,
    key: Union[int, jnp.ndarray],
    *,
    shape0: Optional[int] = None,
    dtype=None,
    cmap=jax.vmap,
):
    """Computes a stochastic estimate of the log-determinate of a matrix using
    the stochastic Lanczos quadrature algorithm.
    """
    if not isinstance(key, jnp.ndarray):
        key = random.PRNGKey(key)

    if callable(mat):
        mat_fn = mat
    else:
        mat_fn = Partial(matmul, mat)
        shape0 = mat.shape[0] if shape0 is None else None
    if shape0 is None:
        msg = "shape0 must be provided if `mat` is callable or has no shape attribute"
        raise ValueError(msg)

    def random_lanczos(k):
        v = random.rademacher(k, (shape0,), dtype=dtype)
        tri, _ = lanczos_tridiag(mat_fn, v, order=order)
        return tri

    key_smpls = random.split(key, n_samples)
    tridiags = cmap(random_lanczos)(key_smpls)  # shape (n_samples, order, order)
    # return scalar estimate (matrix_shape0 used in quadrature)
    return stochastic_logdet_from_lanczos(tridiags, shape0)
