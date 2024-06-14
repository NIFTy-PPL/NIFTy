#!/usr/bin/env python3
# %%
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from numpy.testing import assert_allclose

import nifty8.re as jft

pmp = pytest.mark.parametrize


# %%
if __name__ == "__main__":
    from nifty8.re.multi_grid import indexing as mgi
    from nifty8.re.refine.util import RefinementMatrices
    from functools import partial
    from nifty8.re.refine.util import refinement_matrices

    grid = mgi.HEALPixGrid(nside0=2, depth=2)

    kernel = lambda x, y, axis=0: jnp.exp(-(jnp.linalg.norm(x - y, axis=axis) ** 2))
    coerce_fine_kernel = False
    window_size = np.array((9,))

    # %%
    opt_lin_filter, kernel_sqrt = [], []
    mapped_kernel = partial(kernel, axis=0)
    mapped_kernel = jax.vmap(mapped_kernel, in_axes=(None, -1), out_axes=-1)
    mapped_kernel = jax.vmap(mapped_kernel, in_axes=(-1, None), out_axes=-1)

    grid_at_lvl = grid.at(0)
    # TODO: Children of multiple pixels could be refined jointly requiring a
    # non-unit stride
    pixel_indices = np.mgrid[tuple(slice(0, sz) for sz in grid_at_lvl.shape)]
    g_shape = pixel_indices.shape[1:]
    assert pixel_indices.shape[0] == grid_at_lvl.ndim
    pixel_indices = pixel_indices.reshape(grid_at_lvl.ndim, -1)
    gc = grid_at_lvl.index2coord(pixel_indices)
    gc = gc.reshape(gc.shape[0], np.prod(g_shape))
    cov = mapped_kernel(gc, gc)
    assert cov.shape == (np.prod(g_shape),) * 2
    cov_sqrt0 = jnp.linalg.cholesky(cov)

    for lvl in range(grid.depth):
        grid_at_lvl = grid.at(lvl)
        # TODO: Children of multiple pixels could be refined jointly requiring a
        # non-unit stride
        pixel_indices = np.mgrid[tuple(slice(0, sz) for sz in grid_at_lvl.shape)]
        g_shape = pixel_indices.shape[1:]
        assert pixel_indices.shape[0] == grid_at_lvl.ndim
        pixel_indices = pixel_indices.reshape(grid_at_lvl.ndim, -1)

        # Compute neighbors and map to coordinates outside of vmap to support
        # non-batchable (non-JAX) coordinate transformations
        gc, _ = grid_at_lvl.neighborhood(pixel_indices, window_size)
        gc = grid_at_lvl.index2coord(gc)
        gf = grid_at_lvl.children(pixel_indices)
        gf = grid.at(lvl + 1).index2coord(gf)
        assert gc.shape[0] == gf.shape[0]
        odim = gc.shape[0]
        gc = gc.reshape(odim, np.prod(g_shape), np.prod(window_size))
        gf = gf.reshape(odim, np.prod(g_shape), np.prod(grid_at_lvl.splits))
        coord = jnp.concatenate((gc, gf), axis=-1)
        del gc, gf
        cov = mapped_kernel(coord, coord)
        del coord
        _cs = np.prod(window_size) + np.prod(grid_at_lvl.splits)
        assert cov.shape == (np.prod(g_shape),) + (_cs,) * 2

        olf, fks = jax.vmap(refinement_matrices, in_axes=(0, None, None))(
            cov, np.prod(grid_at_lvl.splits), coerce_fine_kernel
        )
        olf = olf.reshape(g_shape + tuple(grid_at_lvl.splits) + tuple(window_size))
        fks = fks.reshape(g_shape + tuple(grid_at_lvl.splits) * 2)
        opt_lin_filter.append(olf)
        kernel_sqrt.append(fks)

    # %%
    rfm = RefinementMatrices(opt_lin_filter, kernel_sqrt, cov_sqrt0, None)

    # %%
    key = random.PRNGKey(42)
    key, ke = random.split(key)
    excitations = jft.random_like(
        ke,
        [jft.ShapeWithDtype(grid.at(0).shape)]
        + [
            jft.ShapeWithDtype(tuple(grid.at(lvl).shape) + tuple(grid.at(lvl).splits))
            for lvl in range(grid.depth)
        ],
    )

    # %%
    values = rfm.cov_sqrt0 @ excitations[0]
    precision = None

    for lvl in range(grid.depth):
        print(f"{lvl=}")
        olf, fks = rfm.filter[lvl], rfm.propagator_sqrt[lvl]
        exc = excitations[lvl + 1]

        grid_at_lvl = grid.at(lvl)
        # TODO: Children of multiple pixels could be refined jointly requiring a
        # non-unit stride
        pixel_indices = np.mgrid[tuple(slice(0, sz) for sz in grid_at_lvl.shape)]
        g_shape = pixel_indices.shape[1:]
        assert pixel_indices.shape[0] == grid_at_lvl.ndim
        pixel_indices = pixel_indices.reshape(grid_at_lvl.ndim, -1)

        def refine(coarse_values, coarse_idx, olf, fks, rfm_idx, exc):
            c = coarse_values[tuple(coarse_idx)]
            print(f"{c.shape=} {coarse_idx.shape=}")
            o = olf[rfm_idx] if rfm_idx is not None else olf
            f = fks[rfm_idx] if rfm_idx is not None else fks
            print(f"{o.shape=} {f.shape=}")
            refined = jnp.tensordot(o, c, axes=grid_at_lvl.ndim, precision=precision)
            refined += jnp.tensordot(f, exc, axes=grid_at_lvl.ndim, precision=precision)
            print(f"{refined.shape=}")
            return refined

        nbrs_idx, _ = grid_at_lvl.neighborhood(pixel_indices, window_size)
        mapped_refine = jax.vmap(refine, in_axes=(None, 1, None, None, 0, 0))
        olf = olf.reshape((-1,) + olf.shape[-2 * grid_at_lvl.ndim :])
        fks = fks.reshape((-1,) + fks.shape[-2 * grid_at_lvl.ndim :])
        rfm_idx = (
            jnp.arange(np.prod(g_shape)) if rfm.index_map is None else rfm.index_map
        )
        values = mapped_refine(values, nbrs_idx, olf, fks, rfm_idx, exc)
        assert values.shape == g_shape + tuple(grid_at_lvl.splits)
        ax_group = tuple(range(0, len(g_shape) * 2, 2)) + tuple(
            range(1, len(g_shape) * 2, 2)
        )
        values = jnp.transpose(values, ax_group)
        print(f"{values.shape}")
        values = values.reshape(grid.at(lvl + 1).shape)
