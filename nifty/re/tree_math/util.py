#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import jax
from jax import numpy as jnp
from jax.tree_util import tree_map


def _check(v, cut=1e-16):
    return v > cut


@jax.custom_jvp
def _solve(A, X):
    assert A.ndim == 2
    assert (X.ndim == 1) or (X.ndim == 2)
    v, U = jnp.linalg.eigh(A)
    vi = jax.lax.select(_check(v), 1.0 / v, jnp.zeros_like(v))
    res = U.T @ X
    res *= vi[:, jnp.newaxis] if X.ndim == 2 else vi
    return U @ res


@_solve.defjvp
def _solve_jvp(primals, tangents):
    # Note: Makes use of `stable_inverse` directly to enable stable higher order
    # derivatives. This is a tradeoff against saving compute for first order.
    (A, X), (dA, dX) = primals, tangents
    res = _solve(A, X)
    return res, _solve(A, dX - dA @ res)


def solve(A, B, *, matrix_eqn=False, transposed=False):
    """
    Solves the linear system of equations

    .. math ::
        AX = B

    for X, where A is a symmetric positive definite matrix.
    Mapped over the leafs of the input trees and batched over leading axes.
    This implementation features a custom, more stable jvp rule.

    Parameters
    ----------
    A : tree-like structure of jnp.ndarray
        LHS of the equation
    B : tree-like structure of jnp.ndarray
        RHS of the equation
    matrix_eqn : bool
        Specifies whether X and B are matrices or Vectors (default: False)
    transposed: bool
        Solves the transposed system of equations

        .. math ::
            A^T X^T = B^T

        instead. (default: False)

    Returns
    -------
    out : tree-like structure of jnp.ndarray (like B)
        Solution X of the above equation

    """
    sig = "(m,m),(m,n)->(m,n)" if matrix_eqn else "(m,m),(m)->(m)"
    solve_leaf = jnp.vectorize(_solve, signature=sig)
    if transposed:
        A = tree_map(jnp.matrix_transpose, A)
        B = tree_map(jnp.matrix_transpose, B)
    res = tree_map(solve_leaf, A, B)
    if transposed:
        res = tree_map(jnp.matrix_transpose, res)
    return res


def _get_sqrt(v, U):
    vsq = jax.lax.select(_check(v), jnp.sqrt(v), jnp.zeros_like(v))
    return U @ (vsq[:, jnp.newaxis] * U.T)


@jax.custom_jvp
def _sqrtm(M):
    v, U = jnp.linalg.eigh(M)
    return _get_sqrt(v, U)


@_sqrtm.defjvp
def _sqrtm_jvp(M, dM):
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


def sqrtm(M):
    """
    Computes the principal matrix squre root B of a symmetric square matrix M:

    .. math ::
        M = BB

    with B being positive semidefinite and symmetric.
    Mapped over the leafs of the input tree and batched over leading axes.
    This implementation features a custom, more stable jvp rule (only 1st derivative).

    Parameters
    ----------
    M : tree-like structure of jnp.ndarray
        The input matrix/matrices

    Returns
    -------
    B : tree-like structure of jnp.ndarray (like M)
        Principal matrix square root of M
    """
    sig = "(n,n)->(n,n)"
    return tree_map(jnp.vectorize(_sqrtm, signature=sig), M)


def _logm(M):
    v, U = jnp.linalg.eigh(M)
    vlog = jnp.log(v)
    return U @ (vlog[:, jnp.newaxis] * U.T)


def logm(M):
    """
    Computes the matrix logarithm of a symmetric square matrix.
    Input matrix must have strictly positive eigenvalues.
    Mapped over the leafs of the input tree and batched over leading axes.

    Parameters
    ----------
    M : tree-like structure of jnp.ndarray
        The input matrix/matrices

    Returns
    -------
    out : tree-like structure of jnp.ndarray (like M)
        Matrix logarithm of M
    """
    sig = "(n,n)->(n,n)"
    return tree_map(jnp.vectorize(_logm, signature=sig), M)
