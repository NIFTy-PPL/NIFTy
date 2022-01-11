#!/usr/bin/env python3

from functools import partial
from string import ascii_uppercase
from typing import Callable, Optional

from jax import vmap
from jax import numpy as jnp
from jax.lax import conv_general_dilated
import numpy as np

# N - batch dimension
# C - feature dimension of data (channel)
# I - input dimension of kernel
# O - output dimension of kernel
CONV_DIMENSION_NAMES = "".join(el for el in ascii_uppercase if el not in "NCIO")


def _get_cov_from_loc(kernel=None, cov_from_loc=None):
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
    cov_from_loc: Optional[Callable] = None
):
    cov_from_loc = _get_cov_from_loc(kernel, cov_from_loc)
    distances = jnp.asarray(distances)

    n_dim = distances.size
    gc = distances.reshape(n_dim, 1) * jnp.array([-1., 0., 1.])
    gc = jnp.stack(jnp.meshgrid(*gc, indexing="ij"), axis=-1)
    gf = distances.reshape(n_dim, 1) * jnp.array([-0.25, 0.25])
    gf = jnp.stack(jnp.meshgrid(*gf, indexing="ij"), axis=-1)
    # On the GPU a single `cov_from_loc` call is about twice as fast as three
    # separate calls for coarse-coarse, fine-fine and coarse-fine.
    coord = jnp.concatenate(
        (gc.reshape(-1, n_dim), gf.reshape(-1, n_dim)), axis=0
    )
    cov = cov_from_loc(coord, coord)
    cov_ff = cov[-2**n_dim:, -2**n_dim:]
    cov_fc = cov[-2**n_dim:, :-2**n_dim]
    cov_cc_inv = jnp.linalg.inv(cov[:-2**n_dim, :-2**n_dim])

    olf = cov_fc @ cov_cc_inv
    fine_kernel_sqrt = jnp.linalg.cholesky(
        cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
    )

    return olf, fine_kernel_sqrt


def refinement_matrices(
    size0,
    depth,
    distances,
    kernel: Optional[Callable] = None,
    cov_from_loc: Optional[Callable] = None
):
    cov_from_loc = _get_cov_from_loc(kernel, cov_from_loc)

    size0 = np.atleast_1d(size0)
    distances = jnp.atleast_1d(distances)
    if size0.shape != distances.shape:
        ve = (
            f"shape of `size0` {size0.shape} is incompatible with"
            f" shape of `distances` {distances.shape}"
        )
        raise ValueError(ve)
    c0 = [d * jnp.arange(sz, dtype=float) for d, sz in zip(distances, size0)]
    coord0 = jnp.stack(jnp.meshgrid(*c0, indexing="ij"), axis=-1)
    coord0 = coord0.reshape(-1, len(size0))
    cov_sqrt0 = jnp.linalg.cholesky(cov_from_loc(coord0, coord0))

    dist_by_depth = distances * 0.5**jnp.arange(1, depth).reshape(-1, 1)
    olaf = partial(layer_refinement_matrices, cov_from_loc=cov_from_loc)
    opt_lin_filter, kernel_sqrt = vmap(olaf, in_axes=0,
                                       out_axes=(0, 0))(dist_by_depth)
    return opt_lin_filter, (cov_sqrt0, kernel_sqrt)


def refine_conv_general(
    coarse_values, excitations, olf, fine_kernel_sqrt, precision=None
):
    n_dim = np.ndim(coarse_values)
    dim_names = CONV_DIMENSION_NAMES[:n_dim]
    # Introduce an artificial channel dimension for the matrix product
    olf = olf.reshape((2**n_dim, ) + (3, ) * (n_dim - 1) + (1, 3))
    fine_kernel_sqrt = fine_kernel_sqrt.reshape((2**n_dim, ) * 2)
    excitations = excitations.reshape((-1, 2**n_dim))

    conv = partial(
        conv_general_dilated,
        window_strides=(1, ) * n_dim,
        padding="valid",
        dimension_numbers=(
            f"N{dim_names}C", f"O{dim_names}I", f"N{dim_names}C"
        ),
        precision=precision,
    )
    fine = jnp.zeros(tuple(n - 2 for n in coarse_values.shape) + (2**n_dim, ))
    c_shp_n1 = coarse_values.shape[-1]
    c_slc_shp = (1, ) + coarse_values.shape[:-1] + (-1, 3)
    fine = fine.at[..., 0::3, :].set(
        conv(
            coarse_values[..., :c_shp_n1 - c_shp_n1 % 3].reshape(c_slc_shp), olf
        )[0]
    )
    fine = fine.at[..., 1::3, :].set(
        conv(
            coarse_values[...,
                          1:c_shp_n1 - (c_shp_n1 - 1) % 3].reshape(c_slc_shp),
            olf
        )[0]
    )
    fine = fine.at[..., 2::3, :].set(
        conv(
            coarse_values[...,
                          2:c_shp_n1 - (c_shp_n1 - 2) % 3].reshape(c_slc_shp),
            olf
        )[0]
    )

    fine += vmap(jnp.matmul,
                 in_axes=(None, 0))(fine_kernel_sqrt,
                                    excitations).reshape(fine.shape)

    fine = fine.reshape(fine.shape[:-1] + (2, ) * n_dim)
    ax_label = np.arange(2 * n_dim)
    ax_t = [e for els in zip(ax_label[:n_dim], ax_label[n_dim:]) for e in els]
    fine = jnp.transpose(fine, axes=ax_t)

    return fine.reshape(tuple(2 * (n - 2) for n in coarse_values.shape))


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


def correlated_field(xi, distances, kernel, precision=None):
    size0, depth = xi[0].size, len(xi)
    os, (cov_sqrt0, ks) = refinement_matrices(
        size0, depth, distances=distances, kernel=kernel
    )

    fine = cov_sqrt0 @ xi[0]
    for x, olf, ks in zip(xi[1:], os, ks):
        fine = refine(fine, x, olf, ks, precision=precision)
    return fine
