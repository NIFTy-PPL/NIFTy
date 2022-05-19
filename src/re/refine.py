#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
from math import ceil
from string import ascii_uppercase
from typing import Callable, Literal, Optional, Union

from jax import vmap
from jax import numpy as jnp
from jax.lax import conv_general_dilated, dynamic_slice, fori_loop
import numpy as np

NDARRAY = Union[jnp.ndarray, np.ndarray]
# N - batch dimension
# C - feature dimension of data (channel)
# I - input dimension of kernel
# O - output dimension of kernel
CONV_DIMENSION_NAMES = "".join(el for el in ascii_uppercase if el not in "NCIO")


def _assert(assertion):
    if not assertion:
        raise AssertionError()


def _get_cov_from_loc(kernel=None,
                      cov_from_loc=None
                     ) -> Callable[[NDARRAY, NDARRAY], NDARRAY]:
    if cov_from_loc is None and callable(kernel):

        def cov_from_loc_sngl(x, y):
            return kernel(jnp.linalg.norm(x - y))

        cov_from_loc = vmap(
            vmap(cov_from_loc_sngl, in_axes=(None, 0)), in_axes=(0, None)
        )
    else:
        if not callable(cov_from_loc):
            ve = "exactly one of `cov_from_loc` or `kernel` must be set and callable"
            raise ValueError(ve)
    # TODO: benchmark whether using `triu_indices(n, k=1)` and
    # `diag_indices(n)` is advantageous
    return cov_from_loc


def layer_refinement_matrices(
    distances,
    kernel: Optional[Callable] = None,
    cov_from_loc: Optional[Callable] = None,
    *,
    _coarse_size: int = 3,
    _fine_size: int = 2,
    _fine_strategy: Literal["jump", "extend"] = "jump",
    _with_zeros: bool = False,
):
    cov_from_loc = _get_cov_from_loc(kernel, cov_from_loc)
    distances = jnp.asarray(distances)
    # TODO: distances must be a tensor iff _coarse_size > 3
    # TODO: allow different grid sizes for different axis
    csz = int(_coarse_size)  # coarse size
    if _coarse_size % 2 != 1:
        raise ValueError("only odd numbers allowed for `_coarse_size`")
    fsz = int(_fine_size)  # fine size
    if _fine_size % 2 != 0:
        raise ValueError("only even numbers allowed for `_fine_size`")

    ndim = distances.size
    csz_half = int((csz - 1) / 2)
    gc = jnp.arange(-csz_half, csz_half + 1, dtype=float)
    gc = distances.reshape(ndim, 1) * gc
    gc = jnp.stack(jnp.meshgrid(*gc, indexing="ij"), axis=-1)
    if _fine_strategy == "jump":
        gf = jnp.arange(fsz, dtype=float) / fsz - 0.5 + 0.5 / fsz
    elif _fine_strategy == "extend":
        gf = jnp.arange(fsz, dtype=float) / 2 - 0.25 * (fsz - 1)
    else:
        raise ValueError(f"invalid `_fine_strategy`; got {_fine_strategy}")
    gf = distances.reshape(ndim, 1) * gf
    gf = jnp.stack(jnp.meshgrid(*gf, indexing="ij"), axis=-1)
    # On the GPU a single `cov_from_loc` call is about twice as fast as three
    # separate calls for coarse-coarse, fine-fine and coarse-fine.
    coord = jnp.concatenate(
        (gc.reshape(-1, ndim), gf.reshape(-1, ndim)), axis=0
    )
    cov = cov_from_loc(coord, coord)
    cov_ff = cov[-fsz**ndim:, -fsz**ndim:]
    cov_fc = cov[-fsz**ndim:, :-fsz**ndim]
    cov_cc = cov[:-fsz**ndim, :-fsz**ndim]
    cov_cc_inv = jnp.linalg.inv(cov_cc)

    olf = cov_fc @ cov_cc_inv
    # Also see Schur-Complement
    if _with_zeros:
        r = jnp.linalg.norm(gc.reshape(-1, ndim), axis=1)
        r_cutoff = jnp.max(distances) * csz_half
        # dampening is chosen somewhat arbitrarily
        r_dampening = jnp.max(distances)**-ndim
        olf_wgt_sphere = jnp.where(
            r <= r_cutoff, 1.,
            jnp.exp(-r_dampening * jnp.abs(r - r_cutoff)**ndim)
        )
        olf *= olf_wgt_sphere[jnp.newaxis, ...]
        fine_kernel = cov_ff - olf @ cov_cc @ olf.T
    else:
        fine_kernel = cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
    # Implicitly assume a white power spectrum beyond the numerics limit. Use
    # the diagonal as estimate for the magnitude of the variance.
    fine_kernel_fallback = jnp.diag(jnp.abs(jnp.diag(fine_kernel)))
    # Never produce NaNs (https://github.com/google/jax/issues/1052)
    fine_kernel = jnp.where(
        jnp.all(jnp.diag(fine_kernel) > 0.), fine_kernel, fine_kernel_fallback
    )
    fine_kernel_sqrt = jnp.linalg.cholesky(fine_kernel)

    return olf, fine_kernel_sqrt


def refinement_matrices(
    shape0,
    depth,
    distances0,
    kernel: Optional[Callable] = None,
    cov_from_loc: Optional[Callable] = None,
    *,
    _coarse_size: int = 3,
    _fine_size: int = 2,
    _fine_strategy: Literal["jump", "extend"] = "jump",
    **kwargs,
):
    cov_from_loc = _get_cov_from_loc(kernel, cov_from_loc)

    shape0 = np.atleast_1d(shape0)
    distances0 = jnp.atleast_1d(distances0)
    if shape0.shape != distances0.shape:
        ve = (
            f"shape of `shape0` {shape0.shape} is incompatible with"
            f" shape of `distances0` {distances0.shape}"
        )
        raise ValueError(ve)
    c0 = [d * jnp.arange(sz, dtype=float) for d, sz in zip(distances0, shape0)]
    coord0 = jnp.stack(jnp.meshgrid(*c0, indexing="ij"), axis=-1)
    coord0 = coord0.reshape(-1, len(shape0))
    cov_sqrt0 = jnp.linalg.cholesky(cov_from_loc(coord0, coord0))

    if _fine_strategy == "jump":
        dist_by_depth = distances0 / _fine_size**jnp.arange(0, depth
                                                           ).reshape(-1, 1)
    elif _fine_strategy == "extend":
        dist_by_depth = distances0 / 2**jnp.arange(0, depth).reshape(-1, 1)
    else:
        raise ValueError(f"invalid `_fine_strategy`; got {_fine_strategy}")
    olaf = partial(
        layer_refinement_matrices,
        cov_from_loc=cov_from_loc,
        _coarse_size=_coarse_size,
        _fine_size=_fine_size,
        _fine_strategy=_fine_strategy,
        **kwargs
    )
    opt_lin_filter, kernel_sqrt = vmap(olaf, in_axes=0,
                                       out_axes=(0, 0))(dist_by_depth)
    return opt_lin_filter, (cov_sqrt0, kernel_sqrt)


def _vmap_squeeze_first(fun, *args, **kwargs):
    vfun = vmap(fun, *args, **kwargs)

    def vfun_apply(*x):
        return vfun(jnp.squeeze(x[0], axis=0), *x[1:])

    return vfun_apply


def refine_conv_general(
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
    # Introduce an artificial channel dimension for the matrix product
    # TODO: allow different grid sizes for different axis
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
    olf = olf.reshape(
        irreg_shape + (fsz**ndim, ) + (csz, ) * (ndim - 1) + (1, csz)
    )
    fine_kernel_sqrt = fine_kernel_sqrt.reshape(irreg_shape + (fsz**ndim, ) * 2)

    if _fine_strategy == "jump":
        window_strides = (1, ) * ndim
        fine_init_shape = tuple(n - (csz - 1)
                                for n in coarse_values.shape) + (fsz**ndim, )
        fine_final_shape = tuple(
            fsz * (n - (csz - 1)) for n in coarse_values.shape
        )
        convolution_slices = list(range(csz))
    elif _fine_strategy == "extend":
        window_strides = (fsz // 2, ) * ndim
        fine_init_shape = tuple(
            ceil((n - (csz - 1)) / (fsz // 2)) for n in coarse_values.shape
        ) + (fsz**ndim, )
        fine_final_shape = tuple(
            fsz * ceil((n - (csz - 1)) / (fsz // 2))
            for n in coarse_values.shape
        )
        convolution_slices = list(range(0, csz * fsz // 2, fsz // 2))

        if fsz // 2 > csz:
            ve = "extrapolation is not allowed (use `fine_size / 2 <= coarse_size`)"
            raise ValueError(ve)
    else:
        raise ValueError(f"invalid `_fine_strategy`; got {_fine_strategy}")

    if ndim > len(CONV_DIMENSION_NAMES):
        ve = f"convolution for {ndim} dimensions not yet implemented"
        raise ValueError(ve)
    dim_names = CONV_DIMENSION_NAMES[:ndim]
    conv = partial(
        conv_general_dilated,
        window_strides=window_strides,
        padding="valid",
        # channel-last layout is most efficient for vision models (at least in
        # PyTorch)
        dimension_numbers=(
            f"N{dim_names}C", f"O{dim_names}I", f"N{dim_names}C"
        ),
        precision=precision,
    )

    c_shp_n1 = coarse_values.shape[-1]
    c_slc_shp = (1, )
    c_slc_shp += tuple(
        c if i == 1 else csz
        for i, c in zip(irreg_shape, coarse_values.shape[:-1])
    )
    c_slc_shp += (-1, csz)

    fine = jnp.zeros(fine_init_shape)
    PLC = -1 << 31  # integer placeholder outside of the here encountered regimes
    irreg_indices = jnp.stack(
        jnp.meshgrid(
            *[
                jnp.arange(sz) if sz != 1 else jnp.array([PLC])
                for sz in irreg_shape
            ],
            indexing="ij"
        ),
        axis=-1
    )

    def single_refinement_step(i, fine: jnp.ndarray) -> jnp.ndarray:
        irreg_idx = jnp.unravel_index(i, irreg_indices.shape[:-1])
        _assert(
            len(irreg_shape) == len(irreg_indices[irreg_idx]) ==
            len(window_strides)
        )
        fine_init_idx = tuple(
            idx if sz != 1 else slice(None)
            for sz, idx in zip(irreg_shape, irreg_indices[irreg_idx])
        )
        # Make JAX/XLA happy with `dynamic_slice`
        coarse_idx = tuple(
            (ws * idx, csz) if sz != 1 else (0, cend)
            for ws, sz, idx, cend in zip(
                window_strides, irreg_shape, irreg_indices[irreg_idx],
                coarse_values.shape
            )
        )
        coarse_idx_select = partial(
            dynamic_slice,
            start_indices=list(zip(*coarse_idx))[0],
            slice_sizes=list(zip(*coarse_idx))[1]
        )

        olf_at_i = jnp.squeeze(
            olf[fine_init_idx],
            axis=tuple(range(sum(i == 1 for i in irreg_shape)))
        )
        if irreg_shape[-1] == 1 and fine_init_shape[-1] != 1:
            _assert(fine_init_idx[-1] == slice(None))
            # loop over conv channel offsets to apply the filter matrix in a convolution
            for i_f, i_c in enumerate(convolution_slices):
                c = conv(
                    coarse_idx_select(coarse_values)[..., i_c:c_shp_n1 -
                                                     (c_shp_n1 - i_c) %
                                                     csz].reshape(c_slc_shp),
                    olf_at_i
                )[0]
                c = jnp.squeeze(
                    c,
                    axis=tuple(a for a, i in enumerate(irreg_shape) if i != 1)
                )
                toti = fine_init_idx[:-1] + (slice(i_f, None, csz), )
                fine = fine.at[toti].set(c)
        else:
            _assert(
                not isinstance(fine_init_idx[-1], slice) and
                fine_init_idx[-1].ndim == 0
            )
            c = conv(
                coarse_idx_select(coarse_values).reshape(c_slc_shp), olf_at_i
            )[0]
            c = jnp.squeeze(
                c, axis=tuple(a for a, i in enumerate(irreg_shape) if i != 1)
            )
            fine = fine.at[fine_init_idx].set(c)

        return fine

    fine = fori_loop(
        0, np.prod(irreg_indices.shape[:-1]), single_refinement_step, fine
    )

    matmul = partial(jnp.matmul, precision=precision)
    for i in irreg_shape[::-1]:
        if i != 1:
            matmul = vmap(matmul, in_axes=(0, 0))
        else:
            matmul = _vmap_squeeze_first(matmul, in_axes=(None, 0))
    m = matmul(fine_kernel_sqrt, excitations.reshape(fine_init_shape))
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
            filter_coarse = _vmap_squeeze_first(filter_coarse, in_axes=(None, None, 1))
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


def refine_conv(
    coarse_values, excitations, olf, fine_kernel_sqrt, precision=None
):
    fine_m = vmap(
        partial(jnp.convolve, mode="valid", precision=precision),
        in_axes=(None, 0),
        out_axes=0
    )(coarse_values, olf[::-1])
    fine_m = jnp.moveaxis(fine_m, (0, ), (1, ))
    fine_std = vmap(jnp.matmul, in_axes=(None, 0))(
        fine_kernel_sqrt, excitations.reshape(-1, fine_kernel_sqrt.shape[-1])
    )

    return (fine_m + fine_std).ravel()


def refine_loop(
    coarse_values, excitations, olf, fine_kernel_sqrt, precision=None
):
    fine_m = [
        jnp.convolve(coarse_values, o, mode="valid", precision=precision)
        for o in olf[::-1]
    ]
    fine_m = jnp.stack(fine_m, axis=1)
    fine_std = vmap(jnp.matmul, in_axes=(None, 0))(
        fine_kernel_sqrt, excitations.reshape(-1, fine_kernel_sqrt.shape[-1])
    )

    return (fine_m + fine_std).ravel()


def refine_vmap(
    coarse_values, excitations, olf, fine_kernel_sqrt, precision=None
):
    sh0 = coarse_values.shape[0]
    conv = vmap(
        partial(jnp.matmul, precision=precision), in_axes=(None, 0), out_axes=0
    )
    fine_m = jnp.zeros((coarse_values.size - 2, 2))
    fine_m = fine_m.at[0::3].set(
        conv(olf, coarse_values[:sh0 - sh0 % 3].reshape(-1, 3))
    )
    fine_m = fine_m.at[1::3].set(
        conv(olf, coarse_values[1:sh0 - (sh0 - 1) % 3].reshape(-1, 3))
    )
    fine_m = fine_m.at[2::3].set(
        conv(olf, coarse_values[2:sh0 - (sh0 - 2) % 3].reshape(-1, 3))
    )

    fine_std = vmap(jnp.matmul, in_axes=(None, 0))(
        fine_kernel_sqrt, excitations.reshape(-1, fine_kernel_sqrt.shape[-1])
    )

    return (fine_m + fine_std).ravel()


refine = refine_slice
