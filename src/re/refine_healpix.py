#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from math import log2, sqrt
from typing import Callable, Optional, Tuple
import warnings

from jax import vmap
from jax import numpy as jnp
from jax.lax import dynamic_slice_in_dim
import numpy as np

from .refine import _get_cov_from_loc


def get_1st_hp_nbrs_idx(nside, pix, nest: bool = False, dtype=np.uint32):
    from healpy import pixelfunc

    n_nbr = 8

    n_pix = 1 if np.ndim(pix) == 0 else len(pix)
    pix_nbr = np.zeros((n_pix, n_nbr + 1), dtype=int)
    pix_nbr[:, 0] = pix
    nbr = pixelfunc.get_all_neighbours(nside, pix, nest=nest)
    nbr = nbr.reshape(n_nbr, n_pix).T
    pix_nbr[:, 1:] = nbr
    pix_nbr = np.sort(pix_nbr, axis=1)  # Move `-1` to the front

    # Account for unknown neighbors, encoded by -1
    idx_w_invalid, _ = np.nonzero(nbr == -1)
    if idx_w_invalid.size != 0:
        idx_w_invalid = np.unique(idx_w_invalid)
        nbr_invalid = nbr[idx_w_invalid]
        with warnings.catch_warnings():
            wmsg = "invalid value encountered in _get_neigbors"
            warnings.filterwarnings("ignore", message=wmsg)
            # shape of (n_2nd_neighbors, n_idx_w_invalid, n_1st_neighbors)
            nbr2 = pixelfunc.get_all_neighbours(nside, nbr_invalid, nest=nest)
            nbr2 = np.transpose(nbr2, (1, 2, 0))
            nbr2[nbr_invalid == -1] = -1
            nbr2 = nbr2.reshape(idx_w_invalid.size, -1)
        pix_2nbr = np.stack(
            [
                np.setdiff1d(ar1, ar2)[:n_nbr + 1]
                for ar1, ar2 in zip(nbr2, pix_nbr[idx_w_invalid])
            ]
        )
        if np.sum(pix_2nbr == -1):
            # `setdiff1d` should remove all `-1` because we worked with rows in
            # pix_nbr that all contain them
            raise AssertionError()
        pad = max(n_nbr + 1 - pix_2nbr.shape[1], 0)
        pix_2nbr = np.pad(
            pix_2nbr, ((0, 0), (0, pad)), mode="constant", constant_values=-1
        )
        # Select a "random" 2nd neighbor to fill in for the missing 1st order
        # neighbor
        pix_nbr[idx_w_invalid] = np.where(
            pix_nbr[idx_w_invalid] == -1, pix_2nbr, pix_nbr[idx_w_invalid]
        )

    out = np.squeeze(pix_nbr, axis=0) if np.ndim(pix) == 0 else pix_nbr
    return out.astype(dtype)


def get_all_1st_hp_nbrs_idx(nside, nest: bool = False):
    pix = np.arange(12 * nside**2)
    return get_1st_hp_nbrs_idx(nside, pix, nest=nest)


def get_1st_hp_nbrs(nside, pix, nest: bool = False):
    from healpy import pixelfunc

    return np.stack(
        pixelfunc.pix2vec(
            nside, get_1st_hp_nbrs_idx(nside, pix, nest=nest), nest=nest
        ),
        axis=-1
    )


def cov_sqrt(chart, kernel, level: int = 0):
    from healpy import pixelfunc

    if chart.ndim != 2:
        nie = "covariance computation only implemented for 3D HEALPix"
        raise NotImplementedError(nie)

    nside = chart.nside_at(level)
    pix0s = np.stack(
        pixelfunc.pix2vec(nside, np.arange(12 * nside**2), nest=chart.nest),
        axis=-1
    )
    r_rg0 = jnp.mgrid[tuple(slice(s) for s in chart.shape0[1:])]
    pix0s = (
        pix0s[:, np.newaxis, :] *
        chart.nonhp_ind2cart(r_rg0, level)[..., np.newaxis]
    ).reshape(-1, 3)
    cov_from_loc = _get_cov_from_loc(kernel, None)
    # Matrices are symmetrized by JAX, i.e. gradients are projected to the
    # subspace of symmetric matrices (see
    # https://github.com/google/jax/issues/10815)
    fks_sqrt = jnp.linalg.cholesky(cov_from_loc(pix0s, pix0s))

    return fks_sqrt


def _refinement_matrices(
    gc_and_gf,
    kernel: Optional[Callable] = None,
    *,
    coerce_fine_kernel: bool = True,
    _cov_from_loc: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    cov_from_loc = _get_cov_from_loc(kernel, _cov_from_loc)
    gc, gf = gc_and_gf
    n_fsz = gf.shape[0]

    coord = jnp.concatenate((gc, gf), axis=0)
    cov = cov_from_loc(coord, coord)
    cov_ff = cov[-n_fsz:, -n_fsz:]
    cov_fc = cov[-n_fsz:, :-n_fsz]
    cov_cc = cov[:-n_fsz, :-n_fsz]
    cov_cc_inv = jnp.linalg.inv(cov_cc)

    olf = cov_fc @ cov_cc_inv
    # Also see Schur-Complement
    fine_kernel = cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
    if coerce_fine_kernel:
        # TODO: Try to work with NaN to avoid the expensive eigendecomposition;
        # work with nan_to_num?
        # Implicitly assume a white power spectrum beyond the numerics limit.
        # Use the diagonal as estimate for the magnitude of the variance.
        fine_kernel_fallback = jnp.diag(jnp.abs(jnp.diag(fine_kernel)))
        # Never produce NaNs (https://github.com/google/jax/issues/1052)
        # This is expensive but necessary (worse but cheaper:
        # `jnp.all(jnp.diag(fine_kernel) > 0.)`)
        is_pos_def = jnp.all(jnp.linalg.eigvalsh(fine_kernel) > 0)
        fine_kernel = jnp.where(is_pos_def, fine_kernel, fine_kernel_fallback)
        # NOTE, subsequently use the Cholesky decomposition, even though
        # already having computed the eigenvalues, as to get consistent results
        # across platforms
    # Matrices are symmetrized by JAX, i.e. gradients are projected to the
    # subspace of symmetric matrices (see
    # https://github.com/google/jax/issues/10815)
    fine_kernel_sqrt = jnp.linalg.cholesky(fine_kernel)

    return olf, fine_kernel_sqrt


def _vmap_squeeze_first_2ndax(fun, *args, **kwargs):
    vfun = vmap(fun, *args, **kwargs)

    def vfun_apply(*x):
        return vfun(jnp.squeeze(x[0], axis=1), *x[1:])

    return vfun_apply


def refine(
    coarse_values,
    excitations,
    *,
    chart,
    kernel: Callable,
    coerce_fine_kernel: bool = True,
    precision=None,
):
    # TODO: Check performance of 'ring' versus 'nest' alignment
    if chart.ndim not in (1, 2):
        raise ValueError(
            f"invalid dimensions {chart.ndim!r}; expected either 0 or 1"
        )
    coarse_values = coarse_values[:, np.
                                  newaxis] if chart.ndim == 1 else coarse_values
    nside = sqrt(coarse_values.shape[0] / 12)
    lvl = int(log2(nside) - log2(chart.nside0))

    def refine(coarse_values, exc, idx_hp, idx_r):
        # `idx_r` is the left-most radial pixel of the to-be-refined slice
        # Extend `gc` and `gf` radially
        gc, gf = chart.get_coarse_fine_pair((idx_hp, idx_r), lvl)
        olf, fks = _refinement_matrices(
            (gc, gf), kernel=kernel, coerce_fine_kernel=coerce_fine_kernel
        )
        if chart.ndim > 1:
            olf = olf.reshape(
                chart.fine_size**2, chart.fine_size, chart.coarse_size**2,
                chart.coarse_size
            )

        c = coarse_values[chart.hp_neighbors_idx(lvl, idx_hp)]
        if chart.ndim == 2:
            c = dynamic_slice_in_dim(
                coarse_values[chart.hp_neighbors_idx(lvl, idx_hp)],
                idx_r,
                slice_size=chart.coarse_size,
                axis=1
            )
        refined = jnp.tensordot(olf, c, axes=chart.ndim, precision=precision)
        if chart.ndim == 1:
            f_shp = (chart.fine_size**2, )
        else:
            f_shp = (chart.fine_size**2, chart.fine_size)
        refined += jnp.matmul(fks, exc, precision=precision).reshape(f_shp)
        return refined

    pix_hp_idx = jnp.arange(chart.shape_at(lvl)[0])
    if chart.ndim == 1:
        pix_r_off = None
        vrefine = _vmap_squeeze_first_2ndax(
            refine, in_axes=(None, 0, 0, None, 0, 0)
        )
    elif chart.ndim == 2:
        pix_r_off = jnp.arange(chart.shape_at(lvl)[1] - chart.coarse_size + 1)
        # TODO: benchmark swapping these two
        vrefine = vmap(refine, in_axes=(None, 0, None, 0))
        vrefine = vmap(vrefine, in_axes=(None, 0, 0, None))
    else:
        raise AssertionError()
    refined = vrefine(coarse_values, excitations, pix_hp_idx, pix_r_off)
    if chart.ndim == 1:
        refined = refined.ravel()
    elif chart.ndim == 2:
        refined = jnp.transpose(refined, (0, 2, 1, 3))
        n_hp = refined.shape[0] * refined.shape[1]
        n_r = refined.shape[2] * refined.shape[3]
        refined = refined.reshape(n_hp, n_r)
    else:
        raise AssertionError()
    return refined
