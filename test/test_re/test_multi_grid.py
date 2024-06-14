#!/usr/bin/env python3
# %%
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, value_and_grad
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

    for lvl in range(grid.depth):
        grid_at_lvl = grid.at(lvl)
        assert np.all(grid_at_lvl.splits % 2 == 0)
        # TODO: Children of multiple pixels could be refined jointly requiring a
        # non-unit stride
        pixel_indices = np.mgrid[tuple(slice(0, sz) for sz in grid_at_lvl.shape)]
        g_shape = pixel_indices.shape[1:]
        assert pixel_indices.shape[0] == grid_at_lvl.ndim
        pixel_indices = pixel_indices.reshape(grid_at_lvl.ndim, -1)

        ws = window_size if lvl != 0 else g_shape

        # Compute neighbors and map to coordinates outside of vmap to support
        # non-batchable (non-JAX) coordinate transformations
        gc, _ = grid_at_lvl.neighborhood(pixel_indices, ws)
        gc = grid_at_lvl.index2coord(gc)
        gf = grid_at_lvl.children(pixel_indices)
        gf = grid.at(lvl + 1).index2coord(gf)
        assert gc.shape[0] == gf.shape[0]
        odim = gc.shape[0]
        gc = gc.reshape(odim, np.prod(g_shape), np.prod(ws))
        gf = gf.reshape(odim, np.prod(g_shape), np.prod(grid_at_lvl.splits))
        coord = jnp.concatenate((gc, gf), axis=-1)
        del gc, gf
        cov = mapped_kernel(coord, coord)
        del coord
        _cs = np.prod(ws) + np.prod(grid_at_lvl.splits)
        assert cov.shape == (np.prod(g_shape),) + (_cs,) * 2

        olf, ks = jax.vmap(refinement_matrices, in_axes=(0, None, None))(
            cov, np.prod(grid_at_lvl.splits), coerce_fine_kernel
        )
        olf = olf.reshape(g_shape + olf.shape[-2:])
        ks = ks.reshape(g_shape + ks.shape[-2:])
        opt_lin_filter.append(olf)
        kernel_sqrt.append(ks)

    # %%
    rfm = RefinementMatrices(
        opt_lin_filter, kernel_sqrt, None, (None,) * len(opt_lin_filter)
    )
