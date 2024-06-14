#!/usr/bin/env python3
# %%
import sys

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

    grid = mgi.HEALPixGrid(nside0=16, depth=2)
    gridA = grid.at(1)

    kernel = lambda x, y: np.exp(-(jnp.linalg.norm(x - y) ** 2))
    coerce_fine_kernel: bool = False

    def ref_mat(
        kernel: callable,
        grid: mgi.Grid,
        level: int,
        idx: np.ndarray,
        window_size: int = 5,
        *,
        coerce_fine_kernel: bool = False,
    ):
        from nifty8.re.refine.util import refinement_matrices

        grid_at_level = grid.at(level)
        grid_at_next_level = grid.at(level + 1)

        ws = (
            (window_size,) * grid_at_level.ndim
            if isinstance(window_size, int)
            else window_size
        )
        gc = grid_at_level.neighborhood(idx, ws)
        gf = grid_at_level.neighborhood(
            grid_at_level.children(idx), grid_at_level.splits
        )
        print(f"{gc.shape=} {gf.shape=}")
        gc = grid_at_level.index2coord(gc)
        gf = grid_at_next_level.index2coord(gf)
        print(f"{gc.shape=} {gf.shape=}")
        coord = jnp.concatenate((gc, gf), axis=0)
        cov = kernel(coord, coord)
        print(f"{cov.shape=}")

        olf, ks = refinement_matrices(
            cov, np.prod(grid_at_level.splits), coerce_fine_kernel=coerce_fine_kernel
        )
        olf = olf  # TODO: reshape
        ks = ks  # TODO: reshape
        return olf, ks


    window_size = 5

    opt_lin_filter, kernel_sqrt = [], []
    rfm_at = jax.vmap(
        partial(ref_mat, kernel, grid, coerce_fine_kernel=coerce_fine_kernel),
        in_axes=(None, 0),
        out_axes=(0, 0),
    )
    for lvl in range(grid.depth):
        grid_at_lvl = grid.at(lvl)
        shape_lvl = grid_at_lvl.shape
        pixel_indices = []
        for ax in range(grid_at_lvl.ndim):
            stride = grid_at_lvl.splits / 2
            # TODO: continue here
            if int(stride) != stride:
                raise ValueError("`fine_size` must be even")
            stride = int(stride)
            pixel_indices.append(jnp.arange(pad, shape_lvl[ax] - pad, stride))
        pixel_indices = jnp.stack(jnp.meshgrid(*pixel_indices, indexing="ij"), axis=-1)
        shape_filtered_lvl = pixel_indices.shape[:-1]
        pixel_indices = pixel_indices.reshape(-1, grid_at_lvl.ndim)

        olf, ks = rfm_at(lvl, pixel_indices)
        shape_bc_lvl = tuple(
            shape_filtered_lvl[i] if i in chart.irregular_axes else 1
            for i in range(grid_at_lvl.ndim)
        )
        opt_lin_filter.append(olf.reshape(shape_bc_lvl + olf.shape[-2:]))
        kernel_sqrt.append(ks.reshape(shape_bc_lvl + ks.shape[-2:]))

    return RefinementMatrices(
        opt_lin_filter, kernel_sqrt, None, (None,) * len(opt_lin_filter)
    )
