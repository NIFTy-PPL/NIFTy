#!/usr/bin/env python3

from functools import partial
import sys

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import distance_matrix

import nifty8.re as jft
from nifty8.re import refine

pmp = pytest.mark.parametrize


def matern_kernel(distance, scale, cutoff, dof):
    from jax.scipy.special import gammaln
    from scipy.special import kv

    reg_dist = jnp.sqrt(2 * dof) * distance / cutoff
    return scale**2 * 2**(1 - dof) / jnp.exp(
        gammaln(dof)
    ) * (reg_dist)**dof * kv(dof, reg_dist)


scale, cutoff, dof = 1., 80., 3 / 2

x = jnp.logspace(-6, 11, base=jnp.e, num=int(1e+5))
y = matern_kernel(x, scale, cutoff, dof)
y = jnp.nan_to_num(y, nan=0.)
kernel = Partial(jnp.interp, xp=x, fp=y)
inv_kernel = Partial(jnp.interp, xp=y, fp=x)


@pmp("dist", (10., 20., 30., 1e+3))
def test_refinement_matrices_1d(dist, kernel=kernel):
    cov_from_loc = refine._get_cov_from_loc(kernel=kernel)

    coarse_coord = dist * jnp.array([0., 1., 2.])
    fine_coord = coarse_coord[tuple(
        jnp.array(coarse_coord.shape) // 2
    )] + (jnp.diff(coarse_coord) / jnp.array([-4., 4.]))
    cov_ff = cov_from_loc(fine_coord, fine_coord)
    cov_fc = cov_from_loc(fine_coord, coarse_coord)
    cov_cc_inv = jnp.linalg.inv(cov_from_loc(coarse_coord, coarse_coord))

    fine_kernel = cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
    fine_kernel_sqrt_diy = jnp.linalg.cholesky(fine_kernel)
    olf_diy = cov_fc @ cov_cc_inv

    olf, fine_kernel_sqrt = refine.layer_refinement_matrices(dist, kernel)

    assert_allclose(olf, olf_diy)
    assert_allclose(fine_kernel_sqrt, fine_kernel_sqrt_diy)


@pmp("seed", (12, 42, 43, 45))
@pmp("dist", (10., 20., 30., 1e+3))
def test_refinement_1d(seed, dist, kernel=kernel):
    rng = np.random.default_rng(seed)

    refs = (
        refine.refine_conv, refine.refine_conv_general, refine.refine_loop,
        refine.refine_vmap, refine.refine_loop
    )
    cov_from_loc = refine._get_cov_from_loc(kernel=kernel)
    olf, fine_kernel_sqrt = refine.layer_refinement_matrices(dist, kernel)

    main_coord = jnp.linspace(0., 1000., 50)
    cov_sqrt = jnp.linalg.cholesky(cov_from_loc(main_coord, main_coord))
    lvl0 = cov_sqrt @ rng.normal(size=main_coord.shape)
    lvl1_exc = rng.normal(size=(2 * (lvl0.size - 2), ))

    fine_reference = refine.refine(lvl0, lvl1_exc, olf, fine_kernel_sqrt)
    rtol = 6. * jnp.finfo(lvl0.dtype.type).eps
    atol = 60. * jnp.finfo(lvl0.dtype.type).eps
    aallclose = partial(
        assert_allclose, desired=fine_reference, rtol=rtol, atol=atol
    )
    for ref in refs:
        print(f"testing {ref.__name__}", file=sys.stderr)
        aallclose(ref(lvl0, lvl1_exc, olf, fine_kernel_sqrt))


@pmp("dist", (60., 1e+3, (80., 80.), (40., 90.), (1e+2, 1e+3, 1e+4)))
def test_refinement_covariance(dist, kernel=kernel):
    distances0 = np.atleast_1d(dist)
    cf = partial(
        refine.correlated_field,
        distances=distances0,
        kernel=kernel,
    )
    exc_shp = [
        jft.ShapeWithDtype((3, ) * len(distances0)),
        jft.ShapeWithDtype((2, ) * len(distances0))
    ]

    cf_shp = jax.eval_shape(cf, exc_shp)
    assert cf_shp.shape == (2, ) * len(distances0)
    c0 = [
        d * jnp.arange(sz, dtype=distances0.dtype)
        for d, sz in zip(distances0 / 2, cf_shp.shape)
    ]
    pos = jnp.stack(jnp.meshgrid(*c0, indexing="ij"), axis=-1)

    probe = jnp.zeros(pos.shape[:-1])
    indices = np.indices(pos.shape[:-1]).reshape(pos.ndim - 1, -1)

    cf_T = jax.linear_transpose(cf, exc_shp)
    cf_cf_T = lambda x: cf(*cf_T(x))
    cov_empirical = jax.vmap(
        lambda idx: cf_cf_T(probe.at[tuple(idx)].set(1.)).ravel(),
        in_axes=1,
        out_axes=-1
    )(indices)

    p = pos.reshape(-1, pos.shape[-1])
    dist_mat = distance_matrix(p, p)
    cov_truth = kernel(dist_mat)

    assert_allclose(cov_empirical, cov_truth, rtol=1e-13, atol=0.)


@pmp("seed", (12, 42, 43, 45))
@pmp("n_dim", (1, 2, 3, 4, 5))
def test_refinement_nd_shape(seed, n_dim, kernel=kernel):
    rng = np.random.default_rng(seed)

    distances = np.exp(rng.normal(size=(n_dim, )))
    cov_from_loc = refine._get_cov_from_loc(kernel=kernel)
    olf, fine_kernel_sqrt = refine.layer_refinement_matrices(distances, kernel)

    shp_i = 5
    gc = distances.reshape(n_dim, 1) * jnp.linspace(0., 1000., shp_i)
    gc = jnp.stack(jnp.meshgrid(*gc, indexing="ij"), axis=-1).reshape(-1, n_dim)
    cov_sqrt = jnp.linalg.cholesky(cov_from_loc(gc, gc))
    lvl0 = (cov_sqrt @ rng.normal(size=gc.shape[0])).reshape((shp_i, ) * n_dim)
    lvl1_exc = rng.normal(size=tuple(n - 2 for n in lvl0.shape) + (2**n_dim, ))

    fine_reference = refine.refine(lvl0, lvl1_exc, olf, fine_kernel_sqrt)
    assert fine_reference.shape == tuple((2 * (shp_i - 2), ) * n_dim)


if __name__ == "__main__":
    test_refinement_matrices_1d(5.)
    test_refinement_1d(42, 5.)
