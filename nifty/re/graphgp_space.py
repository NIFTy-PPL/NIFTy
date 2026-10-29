#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Philipp Frank

"""The graphgp factor of a correlated field: the points (as a graph) and the
settings of its :class:`~nifty.re.gaussian_process.GraphGP` model, which the
correlated field builds at `finalize`. Requires the optional ``graphgp``
package (and ``graphgp_cuda`` for ``cuda=True``)."""

from dataclasses import field
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .correlated_field import KernelSpace
from .gaussian_process import GraphGP

__all__ = ["GraphGPSpace"]


class GraphGPSpace(KernelSpace):
    """Factor on arbitrary points, generated with graphgp
    (:class:`~nifty.re.gaussian_process.GraphGP`, built by the correlated field).

    Parameters
    ----------
    points : array of shape ``(..., dim)``
        The points; the field has the shape ``points.shape[:-1]``. Not needed
        if `graph` is given (then the shape is ``(n_points,)`` unless `shape`
        is given).
    graph : graphgp.Graph, optional
        A previously built graph of the points (:meth:`build_graph`).
    r_min, r_max, n_bins : float, float, int
        Covariance table (see :class:`~nifty.re.correlated_field.KernelSpace`).
        graphgp holds the covariance constant beyond the last radial bin, so
        `r_max` must exceed the largest separation of the points; by default it
        is the diagonal of their bounding box and `r_min` half the smallest
        neighbour distance.
    n0 : int
        Number of points generated densely before the refinement.
    k : int
        Number of nearest predecessors each point is conditioned on.
    cuda : bool
        Use the graphgp CUDA extension (``graphgp_cuda``, the default); set
        it to ``False`` on machines without it.
    jitter : float
        Relative inflation of the covariance at zero distance.
    shape : tuple, optional
        Shape of the field for a given `graph`.
    radial_bins, k_bins : arrays, optional
        Explicit bins in place of `r_min`, `r_max` and `n_bins`.
    """

    points: Any = field(metadata=dict(static=False))
    graph: Any = field(metadata=dict(static=False))

    def __init__(
        self,
        points=None,
        *,
        graph=None,
        r_min=None,
        r_max=None,
        n_bins=1024,
        n0=512,
        k=16,
        cuda: bool = True,
        jitter=1e-3,
        shape=None,
        radial_bins=None,
        k_bins=None,
    ):
        from .graphgp_sharded import ShardedGraph

        self.cuda = bool(cuda)
        self.n0, self.k = int(n0), int(k)
        if points is None and graph is None:
            raise TypeError("either `points` or `graph` is required")
        if points is not None:
            shape = tuple(np.shape(points)[:-1]) if shape is None else tuple(shape)
            dim = int(np.shape(points)[-1])
        elif isinstance(graph, ShardedGraph):
            shape = (graph.n_shards * graph.block_size,) if shape is None else tuple(shape)
            dim = int(np.shape(graph.ext_points)[-1])
        else:
            shape = (int(graph.points.shape[0]),) if shape is None else tuple(shape)
            dim = int(graph.points.shape[1])
        n_points = graph.n_shards * graph.block_size if isinstance(graph, ShardedGraph) else int(graph.points.shape[0]) if graph is not None else int(np.prod(shape))
        if int(np.prod(shape)) != n_points:
            raise ValueError(f"shape {shape} does not match the {n_points} graph points")
        self.points = points
        self.graph = graph
        if (r_max is None and radial_bins is None) or (r_min is None and (radial_bins is None or k_bins is None)):
            pts = self._coordinates()
            if r_max is None and radial_bins is None:
                r_max = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) * (1.0 + 1e-3)
            if r_min is None and (radial_bins is None or k_bins is None):
                r_min = 0.5 * self._min_neighbour_distance(pts)
        super().__init__(
            shape,
            dim,
            r_min=r_min,
            r_max=r_max,
            n_bins=n_bins,
            jitter=jitter,
            radial_bins=radial_bins,
            k_bins=k_bins,
        )

    def _coordinates(self):
        """The points as a flat ``(n, dim)`` float64 numpy array (any order)."""
        from .graphgp_sharded import ShardedGraph

        if self.points is not None:
            return np.asarray(self.points, dtype=np.float64).reshape(-1, np.shape(self.points)[-1])
        if isinstance(self.graph, ShardedGraph):
            g = self.graph
            return np.asarray(g.ext_points, dtype=np.float64)[:, : g.block_size].reshape(-1, np.shape(g.ext_points)[-1])
        return np.asarray(self.graph.points, dtype=np.float64)

    @staticmethod
    def _min_neighbour_distance(pts, n_sample=2**17, seed=0):
        """Smallest distance between a point and its nearest neighbour, over a
        random sample of the points (the full set for small sets); this sets the
        finest bin of the covariance table (``r_min = 0.5 x``)."""
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        q = pts if pts.shape[0] <= n_sample else pts[np.random.default_rng(seed).choice(pts.shape[0], n_sample, replace=False)]
        d, _ = tree.query(q, k=2)
        d = d[:, 1]
        d = d[d > 0]
        if d.size == 0:
            raise ValueError("could not infer `r_min` from the points; pass it explicitly")
        return float(d.min())

    build_graph = staticmethod(GraphGP.build_graph)

    @property
    def n_points(self):
        return int(np.prod(self.shape))

    @property
    def mesh(self):
        """The mesh of sharded `points` or of a sharded graph, else None."""
        from jax.sharding import NamedSharding

        p = self.points
        if isinstance(p, jax.Array) and isinstance(p.sharding, NamedSharding) and p.sharding.spec and p.sharding.spec[0] is not None:
            return p.sharding.mesh
        return None

    @property
    def gp_kwargs(self) -> dict:
        return dict(n0=self.n0, k=self.k, cuda=self.cuda)

    def points_or_graph(self):
        """What :class:`~nifty.re.gaussian_process.GraphGP` is built from: the
        prebuilt graph if any, else the points (the graph is built lazily, on
        the devices when a mesh is given)."""
        return self.graph if self.graph is not None else self.points
