#!/usr/bin/env python3

from functools import partial

from jax import vmap
from jax import numpy as jnp
from jax.lax import conv_general_dilated
import numpy as np


def layer_refinement_matrices(distances, kernel):
    def cov_from_loc_sngl(x, y):
        return kernel(jnp.linalg.norm(x - y))

    cov_from_loc = vmap(
        vmap(cov_from_loc_sngl, in_axes=(None, 0)), in_axes=(0, None)
    )

    distances = jnp.asarray(distances).reshape(-1, 1)
    n_dim = distances.shape[0]
    coarse_coord = distances * jnp.array([-1., 0., 1.])
    coarse_coord = jnp.stack(
        jnp.meshgrid(*coarse_coord, indexing="ij"), axis=-1
    )
    coarse_coord = coarse_coord.reshape(-1, n_dim)
    fine_coord = distances * jnp.array([-.25, .25])
    fine_coord = jnp.stack(jnp.meshgrid(*fine_coord, indexing="ij"), axis=-1)
    fine_coord = fine_coord.reshape(-1, n_dim)
    cov_ff = cov_from_loc(fine_coord, fine_coord)
    cov_fc = cov_from_loc(fine_coord, coarse_coord)
    cov_cc_inv = jnp.linalg.inv(cov_from_loc(coarse_coord, coarse_coord))

    olf = cov_fc @ cov_cc_inv
    fine_kernel_sqrt = jnp.linalg.cholesky(
        cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
    )

    return olf, fine_kernel_sqrt


def refinement_matrices_alt(size0, depth, distances, kernel):
    def cov_from_loc_sngl(x, y):
        return kernel(jnp.linalg.norm(x - y))

    # TODO: more dimensions
    cov_from_loc = vmap(
        vmap(cov_from_loc_sngl, in_axes=(None, 0)), in_axes=(0, None)
    )

    coord0 = distances * jnp.arange(size0, dtype=float)
    cov_sqrt0 = jnp.linalg.cholesky(cov_from_loc(coord0, coord0))

    dist_by_depth = distances * 0.5**jnp.arange(1, depth)
    opt_lin_filter, kernel_sqrt = vmap(
        partial(layer_refinement_matrices, kernel=kernel),
        in_axes=0,
        out_axes=(0, 0)
    )(dist_by_depth)

    return opt_lin_filter, (cov_sqrt0, kernel_sqrt)


def refinement_matrices(size0, depth, distances, kernel):
    def cov_from_loc_sngl(x, y):
        return kernel(jnp.linalg.norm(x - y))

    cov_from_loc = vmap(
        vmap(cov_from_loc_sngl, in_axes=(None, 0)), in_axes=(0, None)
    )

    def olaf(dist):
        n_dim = dist.size
        cc = dist.reshape(n_dim, 1) * jnp.array([-1., 0., 1.])
        cc = jnp.stack(jnp.meshgrid(*cc, indexing="ij"), axis=-1)
        cf = dist.reshape(n_dim, 1) * jnp.array([-0.25, 0.25])
        cf = jnp.stack(jnp.meshgrid(*cf, indexing="ij"), axis=-1)
        coord = jnp.concatenate(
            (cc.reshape(-1, n_dim), cf.reshape(-1, n_dim)), axis=0
        )
        cov = cov_from_loc(coord, coord)
        cov_ff = cov[-2 * n_dim:, -2 * n_dim:]
        cov_fc = cov[-2 * n_dim:, :-2 * n_dim]
        cov_cc_inv = jnp.linalg.inv(cov[:-2 * n_dim, :-2 * n_dim])

        olf = cov_fc @ cov_cc_inv
        fine_kernel_sqrt = jnp.linalg.cholesky(
            cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
        )

        return olf, fine_kernel_sqrt

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
    opt_lin_filter, kernel_sqrt = vmap(olaf, in_axes=0,
                                       out_axes=(0, 0))(dist_by_depth)
    return opt_lin_filter, (cov_sqrt0, kernel_sqrt)


def refine_conv(coarse_values, excitations, olf, fine_kernel_sqrt):
    fine_m = vmap(
        partial(jnp.convolve, mode="valid"), in_axes=(None, 0), out_axes=0
    )(coarse_values, olf[::-1])
    fine_m = jnp.moveaxis(fine_m, (0, ), (1, ))
    fine_std = vmap(jnp.matmul, in_axes=(None, 0))(
        fine_kernel_sqrt, excitations.reshape(-1, fine_kernel_sqrt.shape[-1])
    )

    return (fine_m + fine_std).ravel()


def refine_loop(coarse_values, excitations, olf, fine_kernel_sqrt):
    fine_m = [jnp.convolve(coarse_values, o, mode="valid") for o in olf[::-1]]
    fine_m = jnp.stack(fine_m, axis=1)
    fine_std = vmap(jnp.matmul, in_axes=(None, 0))(
        fine_kernel_sqrt, excitations.reshape(-1, fine_kernel_sqrt.shape[-1])
    )

    return (fine_m + fine_std).ravel()


def refine_conv_general(coarse_values, excitations, olf, fine_kernel_sqrt):
    olf = olf[..., jnp.newaxis]

    sh0 = coarse_values.shape[0]
    conv = partial(
        conv_general_dilated,
        window_strides=(1, ),
        padding="valid",
        dimension_numbers=("NWC", "OIW", "NWC")
    )
    fine_m = jnp.zeros((coarse_values.size - 2, 2))
    fine_m = fine_m.at[0::3].set(
        conv(coarse_values[:sh0 - sh0 % 3].reshape(1, -1, 3), olf)[0]
    )
    fine_m = fine_m.at[1::3].set(
        conv(coarse_values[1:sh0 - (sh0 - 1) % 3].reshape(1, -1, 3), olf)[0]
    )
    fine_m = fine_m.at[2::3].set(
        conv(coarse_values[2:sh0 - (sh0 - 2) % 3].reshape(1, -1, 3), olf)[0]
    )

    fine_std = vmap(jnp.matmul, in_axes=(None, 0))(
        fine_kernel_sqrt, excitations.reshape(-1, fine_kernel_sqrt.shape[-1])
    )

    return (fine_m + fine_std).ravel()


def refine_vmap(coarse_values, excitations, olf, fine_kernel_sqrt):
    sh0 = coarse_values.shape[0]
    conv = vmap(jnp.matmul, in_axes=(None, 0), out_axes=0)
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


refine = refine_conv


def correlated_field(xi, distances, kernel):
    size0, depth = xi[0].size, len(xi)
    os, (cov_sqrt0, ks) = refinement_matrices(
        size0, depth, distances=distances, kernel=kernel
    )

    fine = cov_sqrt0 @ xi[0]
    for x, olf, ks in zip(xi[1:], os, ks):
        fine = refine(fine, x, olf, ks)
    return fine
