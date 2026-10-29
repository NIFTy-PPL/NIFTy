# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
"""Coordinates of a multigrid level as a point set, optionally sharded."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

__all__ = ["grid_points"]


def grid_points(grid, level: int = -1, *, mesh=None, chunk: int = 2**22):
    """The coordinates of the pixels of a multigrid level, shape ``shape +
    (dim,)`` with ``shape`` the level's shape, in the pixel order of the grid.
    This is the classic point set for a graphgp factor on a semi-structured
    grid (:class:`~nifty.re.GraphGPSpace`), e.g. the finest level of the grid
    an ICR factor would live on.

    Parameters
    ----------
    grid : nifty.re.multi_grid.Grid
    level : int
        Level of the grid (default: the finest).
    mesh : jax.sharding.Mesh, optional
        One-dimensional device mesh: the points are returned as a ``jax.Array``
        sharded along the first grid axis in contiguous blocks (the layout the
        sharded GP models use; the axis must be divisible by the number of
        devices). Without a mesh a numpy array is returned.
    chunk : int
        Pixels per call of the grid's coordinate function (some coordinate
        functions allocate several temporaries per pixel).
    """
    g = grid.at(level)
    shape = tuple(int(s) for s in g.shape)
    idx = np.stack(np.meshgrid(*(np.arange(s, dtype=int) for s in shape), indexing="ij"), axis=0)
    flat = idx.reshape(idx.shape[0], -1)
    f = jax.jit(g.index2coord)
    parts = [np.asarray(f(jnp.asarray(flat[:, i : i + chunk]))) for i in range(0, flat.shape[1], chunk)]
    coords = np.moveaxis(np.concatenate(parts, axis=1), 0, -1).reshape(shape + (-1,))
    if mesh is None:
        return coords
    if len(mesh.axis_names) != 1:
        raise ValueError("mesh must be one-dimensional")
    n = mesh.shape[mesh.axis_names[0]]
    if shape[0] % n:
        raise ValueError(f"the first grid axis ({shape[0]}) is not divisible into {n} blocks")
    return jax.device_put(coords, NamedSharding(mesh, P(mesh.axis_names[0])))
