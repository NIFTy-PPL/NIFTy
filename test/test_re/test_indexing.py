#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import pytest

pytest.importorskip("jax")

import numpy as np
from jax.tree_util import tree_flatten, tree_map

from nifty8.re.multi_grid.grid import FlatGrid, Grid, MGrid, OpenGrid
from nifty8.re.multi_grid.grid_impl import HEALPixGrid

pmp = pytest.mark.parametrize


def _test_grid(grid, *, nbr=None, padding=None, seed):
    rng = np.random.default_rng(seed)

    index = np.mgrid[tuple(slice(s) for s in grid.at(0).shape)]
    vol_prev = grid.at(0).index2volume(index).sum()

    for lvl in range(grid.depth + 1):
        ga = grid.at(lvl)

        # Check for input errors (with complete disregard to the output)
        lo, hi = 0, ga.shape
        if padding is not None:
            lo, hi = padding, np.array(ga.shape) - np.array(padding)
        rand_idx = np.array([rng.integers(lo, hi)]).T
        ga.children(rand_idx) if lvl != grid.depth else None
        nbr = (3,) * ga.ndim if nbr is None else nbr
        ga.neighborhood(rand_idx, nbr)

        # Assert indices are mapped to valid coordinates and vice versa
        index = np.mgrid[tuple(slice(s) for s in ga.shape)]
        coord = ga.index2coord(index)
        # Assert coordinates live in a unit cube
        max_r = np.sqrt(coord.shape[0]) + 1e-12
        np.testing.assert_array_equal(np.linalg.norm(coord, axis=0) <= max_r, True)
        index_roundtrip = ga.coord2index(coord)
        np.testing.assert_array_equal(index_roundtrip, index)

        # Validate that the volume does not grow
        vol = ga.index2volume(index).sum()
        assert vol <= vol_prev
        vol_prev = vol

        # Validate children and parent mappings are consistent
        if lvl < grid.depth:
            children = ga.children(rand_idx)
            parents = grid.at(lvl + 1).parent(children)
            n_extra_dims = parents.ndim - rand_idx.ndim
            bc = (slice(None),) * rand_idx.ndim + (np.newaxis,) * n_extra_dims
            np.testing.assert_array_equal(parents - rand_idx[bc], 0)


@pmp("seed", (12,))
@pmp(
    "shape0, splits",
    [(3, 2), ((3,), (2, 4)), ((2, 3), ((2,) * 2, (2,) * 2)), ((1, 2, 3), ())],
)
def test_base_grid(seed, shape0, splits):
    grid = Grid(shape0=shape0, splits=splits)
    if isinstance(splits, int):
        splits = (splits,)
    assert grid.depth == len(splits)
    idx = np.array([0] * grid.at(0).ndim)

    _test_grid(grid, seed=seed)
    params_orig = []
    for lvl in range(grid.depth + 1):
        g = grid.at(lvl)
        ch = g.children(idx) if lvl != grid.depth else None
        nbrs = g.neighborhood(idx, (3,) * g.ndim)
        params_orig.append((g.size, g.ndim, ch, nbrs))

    if len(splits) > 1:
        sp0, sp1 = splits[:1], splits[1:]
        gn = Grid(shape0=shape0, splits=sp0)
        gn = gn.amend(sp1)
        _test_grid(gn, seed=seed)
        params = []
        for lvl in range(gn.depth + 1):
            g = gn.at(lvl)
            ch = g.children(idx) if lvl != grid.depth else None
            nbrs = g.neighborhood(idx, (3,) * g.ndim)
            params.append((g.size, g.ndim, ch, nbrs))
        assert np.all(tree_flatten(tree_map(np.array_equal, params, params_orig))[0])


@pmp("seed", (12,))
@pmp(
    "shape0,splits,padding",
    [((4,), (2,), (1,)), ((4,), (3,), (1,)), ((4, 9), (1, 2), (0, 2))],
)
@pmp("depth", (2,))
def test_open_grid(seed, shape0, splits, padding, depth):
    g = OpenGrid(shape0=shape0, splits=(splits,) * depth, padding=(padding,) * depth)
    _test_grid(g, nbr=(3,) * g.at(0).ndim, padding=padding, seed=seed)


@pmp("seed", (12,))
@pmp("nside0", [1, 2, 16])
@pmp("depth", [0, 1, 2])
def test_hp_grid(seed, nside0, depth):
    g = HEALPixGrid(nside0=nside0, depth=depth)
    assert g.depth == depth
    _test_grid(g, nbr=(9,), seed=seed)


# TODO more tests and input variety
@pmp("seed", (12,))
def test_mgrid(seed):
    g1 = Grid(shape0=(3,), splits=(2,) * 2)
    g2 = HEALPixGrid(nside0=4, depth=2)
    g3 = Grid(shape0=(3, 5), splits=((1, 2),) * 2)
    g = MGrid(g1, g2, g3)
    _test_grid(g, nbr=(3, 9, 2, 3), seed=seed)


@pmp("seed", (12,))
@pmp(
    "grid",
    [
        Grid(shape0=(3,), splits=(2,) * 3),
        HEALPixGrid(nside0=4, depth=2),
        Grid(shape0=(3, 2), splits=((2, 2),) + ((2, 3),) * 2),
    ],
)
@pmp("ordering", ["serial"])
def test_flat_grid(seed, grid, ordering):
    g = FlatGrid(grid, ordering=ordering)
    _test_grid(g, nbr=(9,) * grid.at(0).ndim, seed=seed)


if __name__ == "__main__":
    test_base_grid(seed=12, shape0=(3, 2), splits=(2, 4))
    test_open_grid(seed=12, shape0=(4, 9), splits=(1, 2), padding=(0, 2), depth=2)
    test_hp_grid(seed=12, nside0=16, depth=2)
    test_mgrid(seed=12)
    test_flat_grid(seed=12, grid=HEALPixGrid(nside0=8, depth=2), ordering="serial")
