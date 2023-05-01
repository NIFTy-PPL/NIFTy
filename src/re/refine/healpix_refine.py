#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import warnings
from functools import partial
from math import log2, sqrt

import jax
import numpy as np
from jax import numpy as jnp
from jax.lax import dynamic_slice_in_dim

from .util import get_cov_from_loc


def get_1st_hp_nbrs_idx(nside, pix, nest: bool = False, dtype=np.int32):
    from healpy import pixelfunc

    n_nbr = 8

    n_pix = 1 if np.ndim(pix) == 0 else len(pix)
    pix_nbr = np.zeros((n_pix, n_nbr + 1), dtype=int)  # can contain `-1`
    pix_nbr[:, 0] = pix
    nbr = pixelfunc.get_all_neighbours(nside, pix, nest=nest)
    nbr = nbr.reshape(n_nbr, n_pix).T
    pix_nbr[:, 1:] = nbr

    # Account for unknown neighbors, encoded by -1
    idx_w_invalid, nbr_w_invalid = np.nonzero(pix_nbr == -1)
    if idx_w_invalid.size != 0:
        uniq_idx_w_invalid = np.unique(idx_w_invalid)
        nbr_invalid = nbr[uniq_idx_w_invalid]
        with warnings.catch_warnings():
            wmsg = "invalid value encountered in _get_neigbors"
            warnings.filterwarnings("ignore", message=wmsg)
            # shape of (n_2nd_neighbors, n_idx_w_invalid, n_1st_neighbors)
            nbr2 = pixelfunc.get_all_neighbours(nside, nbr_invalid, nest=nest)
        nbr2 = np.transpose(nbr2, (1, 2, 0))
        nbr2[nbr_invalid == -1] = -1
        nbr2 = nbr2.reshape(uniq_idx_w_invalid.size, -1)
        n_replace = np.sum(pix_nbr[uniq_idx_w_invalid] == -1, axis=1)
        if np.any(np.diff(n_replace)):
            raise AssertionError()
        n_replace = n_replace[0]
        pix_2nbr = np.stack(
            [
                np.setdiff1d(ar1, ar2)[:n_replace]
                for ar1, ar2 in zip(nbr2, pix_nbr[uniq_idx_w_invalid])
            ]
        )
        if np.sum(pix_2nbr == -1):
            # `setdiff1d` should remove all `-1` because we worked with rows in
            # pix_nbr that all contain them
            raise AssertionError()
        # Select a "random" 2nd neighbor to fill in for the missing 1st order
        # neighbor
        pix_nbr[idx_w_invalid, nbr_w_invalid] = pix_2nbr.ravel()
        if np.sum(pix_nbr == -1):
            raise AssertionError()

    out = np.squeeze(pix_nbr, axis=0) if np.ndim(pix) == 0 else pix_nbr
    return out.astype(dtype)


def get_all_1st_hp_nbrs_idx(nside, nest: bool = False, dtype=np.int32):
    pix = np.arange(12 * nside**2)
    return get_1st_hp_nbrs_idx(nside, pix, nest=nest, dtype=dtype)


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
    assert pix0s.ndim == 2
    pix0s = pix0s.reshape(
        pix0s.shape[:1] + (1, ) * (chart.ndim - 1) + pix0s.shape[1:]
    )
    if chart.ndim > 1:
        # NOTE, this works for arbitrary many dimensions but is probably not
        # what the user wants. In 1D for example, the distances are still
        # computed for a 3D sphere with unit radius.
        r_rg0 = jnp.mgrid[tuple(slice(s) for s in chart.shape_at(level)[1:])]
        r_rg0 = jnp.asarray(chart.nonhp_ind2cart(r_rg0, level))[..., np.newaxis]
        pix0s = (pix0s * r_rg0).reshape(-1, 3)
    cov_from_loc = get_cov_from_loc(kernel, None)
    # Matrices are symmetrized by JAX, i.e. gradients are projected to the
    # subspace of symmetric matrices (see
    # https://github.com/google/jax/issues/10815)
    fks_sqrt = jnp.linalg.cholesky(cov_from_loc(pix0s, pix0s))

    return fks_sqrt


@partial(jax.jit, static_argnames=("chart", "precision"))
def refine(
    coarse_values,
    excitations,
    olf,
    fks,
    index_map,
    *,
    chart,
    precision=None,
):
    # TODO: Check performance of 'ring' versus 'nest' alignment
    if chart.ndim != 2:
        nie = f"invalid dimensions {chart.ndim!r}; expected `2`"
        raise NotImplementedError(nie)
    coarse_values = coarse_values[:, np.
                                  newaxis] if chart.ndim == 1 else coarse_values
    nside = sqrt(coarse_values.shape[0] / 12)
    lvl = int(log2(nside) - log2(chart.nside0))

    def refine(coarse_values, exc, idx_hp, idx_r, olf, fks, im):
        c = dynamic_slice_in_dim(
            coarse_values[chart.hp_neighbors_idx(lvl, idx_hp)],
            idx_r,
            slice_size=chart.coarse_size,
            axis=1
        )
        f_shp = (chart.fine_size**2, chart.fine_size)
        o = olf[im] if im is not None else olf
        refined = jnp.tensordot(o, c, axes=chart.ndim, precision=precision)
        f = fks[im] if im is not None else fks
        refined += jnp.matmul(f, exc, precision=precision).reshape(f_shp)
        return refined

    pix_hp_idx = jnp.arange(chart.shape_at(lvl)[0])
    pix_r_off = jnp.arange(chart.shape_at(lvl)[1] - chart.coarse_size + 1)
    # TODO: benchmark swapping these two
    off = index_map is not None
    vrefine = jax.vmap(
        refine, in_axes=(None, 0, None, 0, 0 + off, 0 + off, None)
    )
    in_axes = (None, 0, 0, None)
    in_axes += (0, 0, None) if index_map is None else (None, None, 0)
    vrefine = jax.vmap(vrefine, in_axes=in_axes)
    refined = vrefine(
        coarse_values, excitations, pix_hp_idx, pix_r_off, olf, fks, index_map
    )
    refined = jnp.transpose(refined, (0, 2, 1, 3))
    n_hp = refined.shape[0] * refined.shape[1]
    n_r = refined.shape[2] * refined.shape[3]
    return refined.reshape(n_hp, n_r)
