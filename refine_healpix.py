#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
from math import ceil
from typing import Callable, Iterable, Literal, Optional, Tuple

from healpy import pixelfunc
import numpy as np

from jax import vmap
import jax
from jax import numpy as jnp
from jax.lax import dynamic_slice
from nifty8.re.refine import _get_cov_from_loc

# %%


def get_1st_hp_nbrs_idx(nside, pix, nest: bool = False):
    n_nbr = 8

    n_pix = 1 if np.ndim(pix) == 0 else len(pix)
    pix_nbr = np.empty((n_pix, n_nbr + 1), dtype=int)
    pix_nbr[:, 0] = pix
    nbr = pixelfunc.get_all_neighbours(nside, pix, nest=nest)
    nbr = nbr.reshape(n_nbr, n_pix)
    pix_nbr[:, 1:] = nbr.T

    # Account for unknown neighbors, encoded by -1
    # TODO: capture warning
    # TODO: only do this for affected pixels
    nbr2 = pixelfunc.get_all_neighbours(nside, nbr, nest=nest)
    nbr2 = nbr2.reshape(n_nbr, n_nbr, n_pix)
    nbr2.T[nbr.T == -1] = -1

    n_2nbr = np.prod(nbr2.shape[:-1])
    setdiffer1d = partial(jnp.setdiff1d, size=n_nbr + 2, fill_value=-1)
    pix_2nbr = jax.vmap(setdiffer1d, (1, 1),
                        0)(nbr2.reshape(n_2nbr, n_pix), nbr)
    # If there is a -1 in there, it will be at the first location
    pix_2nbr = pix_2nbr[:, 1:]

    # Select a "random" 2nd neighbor to fill in for the missing 1st order
    # neighbor
    pix_nbr = np.sort(pix_nbr, axis=1)
    pix_nbr = np.where(pix_nbr != -1, pix_nbr, pix_2nbr)
    return np.squeeze(pix_nbr, axis=0) if np.ndim(pix) == 0 else pix_nbr


def get_1st_hp_nbrs(nside, pix, nest: bool = False):
    return np.array(
        pixelfunc.pix2vec(
            nside, get_1st_hp_nbrs_idx(nside, pix, nest=nest), nest=nest
        )
    )


# %%
nside = 256
pix = 0

get_1st_hp_nbrs(nside, pix)


# %%
def _coordinate_pixel_refinement_matrices(
    level: int,
    pixel_index: Optional[Iterable[int]] = None,
    kernel: Optional[Callable] = None,
    *,
    nest: bool = False,
    coerce_fine_kernel: bool = True,
    _cov_from_loc: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    cov_from_loc = _get_cov_from_loc(kernel, _cov_from_loc)
    if pixel_index is None:
        pixel_index = (0, 0)
    pixel_index = jnp.asarray(pixel_index)
    n_fsz = 4  # `n_csz = 9`

    gc = get_1st_hp_nbrs(2**level, pixel_index[0], nest=nest)
    gc = gc.T
    pi = pixel_index[0]
    pi_nest = pixelfunc.ring2nest(2**level, pi) if nest is False else pi
    gf = np.array(
        pixelfunc.pix2vec(2**level, 4 * pi_nest + jnp.arange(0, 4), nest=True)
    )
    gf = gf.T
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


# %%
def _vmap_squeeze_first(fun, *args, **kwargs):
    vfun = vmap(fun, *args, **kwargs)

    def vfun_apply(*x):
        return vfun(jnp.squeeze(x[0], axis=0), *x[1:])

    return vfun_apply


def refine_slice(
    coarse_values,
    excitations,
    olf,
    fine_kernel_sqrt,
    precision=None,
    _coarse_size: int = 3,
    _fine_size: int = 2,
    _fine_strategy: Literal["jump", "extend"] = "jump",
):
    ndim = np.ndim(coarse_values)
    csz = int(_coarse_size)  # coarse size
    if _coarse_size % 2 != 1:
        raise ValueError("only odd numbers allowed for `_coarse_size`")
    fsz = int(_fine_size)  # fine size
    if _fine_size % 2 != 0:
        raise ValueError("only even numbers allowed for `_fine_size`")

    if olf.shape[:-2] != fine_kernel_sqrt.shape[:-2]:
        ve = (
            "incompatible optimal linear filter (`olf`) and `fine_kernel_sqrt` shapes"
            f"; got {olf.shape} and {fine_kernel_sqrt.shape}"
        )
        raise ValueError(ve)
    if olf.ndim > 2:
        irreg_shape = olf.shape[:-2]
    elif olf.ndim == 2:
        irreg_shape = (1, ) * ndim
    else:
        ve = f"invalid shape of optimal linear filter (`olf`); got {olf.shape}"
        raise ValueError(ve)
    olf = olf.reshape(irreg_shape + (fsz**ndim, ) + (csz, ) * ndim)
    fine_kernel_sqrt = fine_kernel_sqrt.reshape(irreg_shape + (fsz**ndim, ) * 2)

    if _fine_strategy == "jump":
        window_strides = (1, ) * ndim
        fine_init_shape = tuple(n - (csz - 1)
                                for n in coarse_values.shape) + (fsz**ndim, )
        fine_final_shape = tuple(
            fsz * (n - (csz - 1)) for n in coarse_values.shape
        )
    elif _fine_strategy == "extend":
        window_strides = (fsz // 2, ) * ndim
        fine_init_shape = tuple(
            ceil((n - (csz - 1)) / (fsz // 2)) for n in coarse_values.shape
        ) + (fsz**ndim, )
        fine_final_shape = tuple(
            fsz * ceil((n - (csz - 1)) / (fsz // 2))
            for n in coarse_values.shape
        )

        if fsz // 2 > csz:
            ve = "extrapolation is not allowed (use `fine_size / 2 <= coarse_size`)"
            raise ValueError(ve)
    else:
        raise ValueError(f"invalid `_fine_strategy`; got {_fine_strategy}")

    def matmul_with_window_into(x, y, idx):
        return jnp.tensordot(
            x,
            dynamic_slice(y, idx, slice_sizes=(csz, ) * ndim),
            axes=ndim,
            precision=precision
        )

    filter_coarse = matmul_with_window_into
    corr_fine = partial(jnp.matmul, precision=precision)
    for i in irreg_shape[::-1]:
        if i != 1:
            filter_coarse = vmap(filter_coarse, in_axes=(0, None, 1))
            corr_fine = vmap(corr_fine, in_axes=(0, 0))
        else:
            filter_coarse = _vmap_squeeze_first(
                filter_coarse, in_axes=(None, None, 1)
            )
            corr_fine = _vmap_squeeze_first(corr_fine, in_axes=(None, 0))

    cv_idx = np.mgrid[tuple(
        slice(None, sz - csz + 1, ws)
        for sz, ws in zip(coarse_values.shape, window_strides)
    )]
    fine = filter_coarse(olf, coarse_values, cv_idx)

    m = corr_fine(fine_kernel_sqrt, excitations.reshape(fine_init_shape))
    rm_axs = tuple(
        ax for ax, i in enumerate(m.shape[len(irreg_shape):], len(irreg_shape))
        if i == 1
    )
    fine += jnp.squeeze(m, axis=rm_axs)

    fine = fine.reshape(fine.shape[:-1] + (fsz, ) * ndim)
    ax_label = np.arange(2 * ndim)
    ax_t = [e for els in zip(ax_label[:ndim], ax_label[ndim:]) for e in els]
    fine = jnp.transpose(fine, axes=ax_t)

    return fine.reshape(fine_final_shape)
