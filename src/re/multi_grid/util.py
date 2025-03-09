#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Philipp Frank

import operator
import jax
from functools import reduce
from jax import numpy as jnp
import numpy as np


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
    # Note: Makes use of `solve` directly to enable stable higher order
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


def nested_index_to_flatindex(coords, base_shape, splits):
    """
    Convert multi-dimensional coordinates to a single nested index,
    where each dimension i has:
        - an integer base_shape[i] for the "coarse" part
        - a refinement base splits[i]
        - depth refinement levels
    so that coords[i] is in the range [0, base_shape[i]*splits[i]^depth).

    Parameters
    ----------
    coords : array_like of shape (d,...)
        Coordinates (X_0, ..., X_{d-1}).
    base_shape : array_like of shape (d,)
        Base shape (M_0, ..., M_{d-1}).
    splits : tuple of array_like of shape (depth,) of shape (d,)
        The refinement base b_i for each dimension at each level.

    Returns
    -------
    idx : int or array_like of shape (...)
        The single nested index corresponding to coords.
    """
    coords = jnp.asarray(coords)
    splits = tuple(splits)
    d = splits[0].size
    depth = len(splits)
    isscalar = coords.ndim == 1
    if isscalar:
        coords = coords[:, jnp.newaxis]

    digits = jnp.zeros((d, depth + 1) + coords.shape[1:], dtype=coords.dtype)
    # Split each X_i into ground part (range M_i) and remainder (range b_i^L).
    if depth > 0:
        fct = reduce(operator.mul, splits)
        ground_part = (
            coords // fct[(slice(None),) + (jnp.newaxis,) * (coords.ndim - 1)]
        )  # floor-division
        remainder = coords % fct[(slice(None),) + (jnp.newaxis,) * (coords.ndim - 1)]
    else:
        ground_part = coords
        remainder = jnp.zeros_like(coords)
    # Fill plane=0 (ground-level digit), must be < M_i
    digits = digits.at[:, 0].set(ground_part)
    # Expand the remainder in base b_i for each dimension
    tmp = remainder.copy()
    for i in range(d):
        for level in range(1, depth + 1):
            # Extract next digit in base b_i
            digits = digits.at[i, level].set(tmp[i] % splits[level - 1][i])
            tmp = tmp.at[i].set(tmp[i] // splits[level - 1][i])

    # Precompute "plane base" for each plane:
    plane_bases = (base_shape,) + splits
    plane_base_products = [np.prod(pb) for pb in plane_bases]
    # Build the index by accumulating the digits
    idx = jnp.zeros(coords.shape[1:], dtype=coords.dtype)
    multiplier = 1
    for plane in range(depth + 1):
        plane_base = plane_bases[plane]
        D_plane = 0
        factor = 1
        for i in range(d):
            D_plane += digits[i, plane] * factor
            factor *= plane_base[i]
        idx += D_plane * multiplier
        multiplier *= plane_base_products[plane]
    if isscalar:
        return idx[0]
    return idx


def nested_flatindex_to_index(idx, base_shape, splits):
    """
    Invert the nested index back to multi-dimensional coordinates,
    for the case of dimension i in [0..d-1] having base_shape[i],
    refinement base splits[level][i] at each level.

    Parameters
    ----------
    idx : int or array_like of shape (d,...)
        The single nested index to decode.
    base_shape : array_like of shape (d,)
        Base shape (M_0, ..., M_{d-1}).
    splits : tuple of array_like of shape (depth,) of shape (d,)
        The refinement base b_i for each dimension at each level.

    Returns
    -------
    coords : ndarray of shape (d,...)
        The multi-dimensional coordinates (X_0, ..., X_{d-1}).
    """
    splits = tuple(splits)
    isscalar = isinstance(idx, int) or idx.ndim == 0
    if isscalar:
        idx = jnp.array(
            [
                idx,
            ]
        )
    d = splits[0].size
    depth = len(splits)

    plane_bases = (base_shape,) + splits
    plane_base_products = [np.prod(pb) for pb in plane_bases]

    # Extract plane-by-plane digits from idx
    digits = jnp.zeros((d, depth + 1) + idx.shape, dtype=idx.dtype)
    tmp = idx.copy()
    for plane in range(depth + 1):
        D_plane = tmp % plane_base_products[plane]
        tmp //= plane_base_products[plane]
        pb = plane_bases[plane]
        for i in range(d):
            digits = digits.at[i, plane].set(D_plane % pb[i])
            D_plane //= pb[i]

    # digits[i, 0] is the ground-level digit in base M_i
    # digits[i, level>0] are the base-b_i digits for that dimension
    coords = jnp.zeros((d,) + idx.shape, dtype=digits.dtype)
    # Reconstruct coords[i] = ground_i * b_i^depth + remainder_i
    fct = reduce(operator.mul, splits)
    for i in range(d):
        ground_i = digits[i, 0]
        remainder_i = jnp.zeros(idx.shape, dtype=digits.dtype)
        factor = 1
        for level in range(1, depth + 1):
            remainder_i += digits[i, level] * factor
            factor *= splits[level - 1][i]
        coords = coords.at[i].set(ground_i * fct[i] + remainder_i)

    if isscalar:
        return coords[:, 0]
    return coords
