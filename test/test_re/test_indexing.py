#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import numpy as np
import pytest
from jax.tree_util import tree_flatten, tree_map

from nifty8.re.multi_grid.grid import FlatGrid, Grid, MGrid
from nifty8.re.multi_grid.grid_impl import HEALPixGrid

pmp = pytest.mark.parametrize


def _test_grid(grid, id=None, nbr=None, *, seed):
    rng = np.random.default_rng(seed)

    params = []
    for lvl in range(grid.depth + 1):
        ga = grid.at(lvl)
        sz = ga.size
        nd = ga.ndim
        id = np.array([rng.integers(0, 1 + 1)] * nd) if id is None else id
        ch = ga.children(id) if lvl != grid.depth else None
        nbr = (3,) * nd if nbr is None else nbr
        nbrs = ga.neighborhood(id, nbr)
        params.append((sz, nd, ch, nbrs))

        if lvl != 0:
            g_ap = grid.at(lvl - 1)
            children = g_ap.children(id)
            parents = ga.parent(children)
            nextra = parents.ndim - id.ndim
            sl = (slice(None),) * id.ndim + (np.newaxis,) * nextra
            assert np.all(parents == id[sl])

    return params


@pmp("seed", (12,))
@pmp(
    "shape0, splits",
    [(3, 2), ((3,), (2, 4)), ((2, 3), ((2,) * 2, (2,) * 2)), ((1, 2, 3), ())],
)
def test_grid(seed, shape0, splits):
    g = Grid(shape0=shape0, splits=splits)
    if isinstance(splits, int):
        splits = (splits,)
    assert g.depth == len(splits)

    params = _test_grid(g, seed=seed)

    if len(splits) > 1:
        sp0 = splits[:1]
        sp1 = splits[1:]
        gn = Grid(shape0=shape0, splits=sp0)
        gn = gn.amend(sp1)
        paramsn = _test_grid(gn, seed=seed)
        valid = tree_map(np.array_equal, params, paramsn)
        assert np.all(tree_flatten(valid)[0])


@pmp("seed", (12,))
@pmp("nside0", [1, 2, 16, 128])
@pmp("depth", [0, 1, 4, 7])
def test_hp_grid(seed, nside0, depth):
    g = HEALPixGrid(nside0=nside0, depth=depth)

    assert g.depth == depth
    gn = HEALPixGrid(nside0=nside0, depth=depth + 2)
    paramsn = _test_grid(gn, nbr=(9,), seed=seed)

    g = g.amend(added_depth=2)
    params = _test_grid(g, nbr=(9,), seed=seed)
    valid = tree_map(np.array_equal, params, paramsn)
    assert np.all(tree_flatten(valid)[0])


# TODO more tests and input variety
@pmp("seed", (12,))
def test_ogrid(seed):
    g1 = Grid(shape0=(3,), splits=(2,) * 3)
    g2 = HEALPixGrid(nside0=4, depth=3)
    g3 = Grid(shape0=(3, 5), splits=((1, 2),) * 3)
    g = MGrid(g1, g2, g3)
    _test_grid(g, nbr=(3, 9, 2, 3), seed=seed)


@pmp("seed", (12,))
@pmp(
    "grid",
    [
        Grid(shape0=(3,), splits=(2,) * 3),
        HEALPixGrid(nside0=8, depth=2),
        Grid(shape0=(3, 2), splits=((2,) * 2,) + ((2, 3),) * 4),
    ],
)
@pmp("ordering", ["serial"])
def test_flat_grid(seed, grid, ordering):
    g = FlatGrid(grid, ordering=ordering)

    _test_grid(g, nbr=(9,) * grid.at(0).ndim, seed=seed)
