#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
from math import ceil
from string import ascii_uppercase
from typing import Callable, Literal, Optional, Union
from warnings import warn

from jax import vmap
from jax import numpy as jnp
from jax.lax import conv_general_dilated
import numpy as np

NDARRAY = Union[jnp.ndarray, np.ndarray]
# N - batch dimension
# C - feature dimension of data (channel)
# I - input dimension of kernel
# O - output dimension of kernel
CONV_DIMENSION_NAMES = "".join(el for el in ascii_uppercase if el not in "NCIO")


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
    olf = olf.reshape((fsz**ndim, ) + (csz, ) * (ndim - 1) + (1, csz))
    fine_kernel_sqrt = fine_kernel_sqrt.reshape((fsz**ndim, ) * 2)

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
    fine = jnp.zeros(fine_init_shape)
    c_shp_n1 = coarse_values.shape[-1]
    c_slc_shp = (1, ) + coarse_values.shape[:-1] + (-1, csz)
    for i_f, i_c in enumerate(convolution_slices):
        fine = fine.at[..., i_f::csz, :].set(
            conv(
                coarse_values[..., i_c:c_shp_n1 -
                              (c_shp_n1 - i_c) % csz].reshape(c_slc_shp), olf
            )[0]
        )

    excitations = excitations.reshape((-1, fsz**ndim))
    fine += vmap(jnp.matmul,
                 in_axes=(None, 0))(fine_kernel_sqrt,
                                    excitations).reshape(fine.shape)

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


refine = refine_conv_general


def get_refinement_shapewithdtype(
    shape0: Union[int, tuple],
    depth: int,
    dtype=None,
    *,
    _coarse_size: int = 3,
    _fine_size: int = 2,
    _fine_strategy: Literal["jump", "extend"] = "jump",
):
    from .forest_util import ShapeWithDtype

    if depth < 0:
        raise ValueError(f"invalid `depth`; got {depth!r}")
    csz = int(_coarse_size)  # coarse size
    fsz = int(_fine_size)  # fine size

    shape0 = (shape0, ) if isinstance(shape0, int) else shape0
    ndim = len(shape0)
    exc_shp = [shape0]
    if depth > 0:
        if _fine_strategy == "jump":
            exc_shp += [
                tuple(el - (csz - 1) for el in exc_shp[0]) + (fsz**ndim, )
            ]
        elif _fine_strategy == "extend":
            exc_shp += [
                tuple(ceil((el - (csz - 1)) / (fsz // 2))
                      for el in exc_shp[0]) + (fsz**ndim, )
            ]
        else:
            raise ValueError(f"invalid `_fine_strategy`; got {_fine_strategy}")
    for lvl in range(1, depth):
        if _fine_strategy == "jump":
            exc_lvl = tuple(fsz * el - (csz - 1)
                            for el in exc_shp[-1][:-1]) + (fsz**ndim, )
        elif _fine_strategy == "extend":
            exc_lvl = tuple(
                ceil((fsz * el - (csz - 1)) / (fsz // 2))
                for el in exc_shp[-1][:-1]
            ) + (fsz**ndim, )
        else:
            raise AssertionError()
        if any(el <= 0 for el in exc_lvl):
            ve = (
                f"`shape0` ({shape0}) with `depth` ({depth}) yield an"
                f" invalid shape ({exc_lvl}) at level {lvl}"
            )
            raise ValueError(ve)
        exc_shp += [exc_lvl]

    exc_shp = list(map(partial(ShapeWithDtype, dtype=dtype), exc_shp))
    return exc_shp


def get_fixed_power_correlated_field(
    shape0: Union[int, tuple],
    distances0,
    depth: int,
    kernel: Callable,
    dtype=None,
    *,
    precision=None,
    **kwargs,
):
    cf = partial(
        correlated_field,
        distances0=distances0,
        kernel=kernel,
        precision=precision,
        **kwargs,
    )
    exc_swd = get_refinement_shapewithdtype(
        shape0,
        depth=depth,
        dtype=dtype,
        **kwargs,
    )
    return cf, exc_swd


def correlated_field(xi, distances0, kernel, **kwargs):
    precision = kwargs.pop("precision", None)

    shape0, depth = xi[0].shape, len(xi) - 1
    os, (cov_sqrt0, ks) = refinement_matrices(
        shape0, depth, distances0=distances0, kernel=kernel, **kwargs
    )

    fine = (cov_sqrt0 @ xi[0].ravel()).reshape(xi[0].shape)
    for x, olf, k in zip(xi[1:], os, ks):
        fine = refine(fine, x, olf, k, precision=precision, **kwargs)
    return fine
