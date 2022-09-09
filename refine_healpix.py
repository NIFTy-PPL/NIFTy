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
nside = 2
pix = 0

pix_neighbors = pixelfunc.get_all_neighbours(nside, pix)
np.array(pixelfunc.pix2vec(nside, pix_neighbors))


# %%
def _coordinate_pixel_refinement_matrices(
    chart,
    level: int,
    pixel_index: Optional[Iterable[int]] = None,
    kernel: Optional[Callable] = None,
    *,
    coerce_fine_kernel: bool = True,
    _cov_from_loc: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    cov_from_loc = _get_cov_from_loc(kernel, _cov_from_loc)
    csz = int(chart.coarse_size)  # coarse size
    if csz % 2 != 1:
        raise ValueError("only odd numbers allowed for `_coarse_size`")
    fsz = int(chart.fine_size)  # fine size
    if fsz % 2 != 0:
        raise ValueError("only even numbers allowed for `_fine_size`")
    ndim = chart.ndim
    if pixel_index is None:
        pixel_index = (0, ) * ndim
    pixel_index = jnp.asarray(pixel_index)
    if pixel_index.size != ndim:
        ve = f"`pixel_index` has {pixel_index.size} dimensions but `chart` has {ndim}"
        raise ValueError(ve)

    csz_half = int((csz - 1) / 2)
    gc = jnp.arange(-csz_half, csz_half + 1, dtype=float)
    gc = jnp.ones((ndim, 1)) * gc
    gc = jnp.stack(jnp.meshgrid(*gc, indexing="ij"), axis=-1)
    if chart.fine_strategy == "jump":
        gf = jnp.arange(fsz, dtype=float) / fsz - 0.5 + 0.5 / fsz
    elif chart.fine_strategy == "extend":
        gf = jnp.arange(fsz, dtype=float) / 2 - 0.25 * (fsz - 1)
    else:
        raise ValueError(f"invalid `_fine_strategy`; got {chart.fine_strategy}")
    gf = jnp.ones((ndim, 1)) * gf
    gf = jnp.stack(jnp.meshgrid(*gf, indexing="ij"), axis=-1)
    # On the GPU a single `cov_from_loc` call is about twice as fast as three
    # separate calls for coarse-coarse, fine-fine and coarse-fine.
    coord = jnp.concatenate(
        (gc.reshape(-1, ndim), gf.reshape(-1, ndim)), axis=0
    )
    coord = chart.ind2cart((coord + pixel_index.reshape((1, ndim))).T, level)
    coord = jnp.stack(coord, axis=-1)
    cov = cov_from_loc(coord, coord)
    cov_ff = cov[-fsz**ndim:, -fsz**ndim:]
    cov_fc = cov[-fsz**ndim:, :-fsz**ndim]
    cov_cc = cov[:-fsz**ndim, :-fsz**ndim]
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
