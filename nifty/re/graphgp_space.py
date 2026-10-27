#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Philipp Frank

"""graphgp (nearest-neighbour Vecchia Gaussian processes on a k-d tree
ordering) as a factor of a correlated field. Requires the optional ``graphgp``
package (and ``graphgp_cuda`` for ``cuda=True``)."""

import importlib.util
from dataclasses import field
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .correlated_field import KernelSpace

__all__ = ["GraphGPSpace"]


def _cuda_available() -> bool:
    try:
        return jax.default_backend() == "gpu" and importlib.util.find_spec("graphgp_cuda") is not None
    except Exception:
        return False


class GraphGPSpace(KernelSpace):
    """Factor on arbitrary points, generated with graphgp.

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
    cuda : bool, optional
        Use the graphgp CUDA extension (default: if a GPU and the extension
        are available).
    jitter : float
        Relative inflation of the covariance at zero distance.
    shape : tuple, optional
        Shape of the field for a given `graph`.
    radial_bins, k_bins : arrays, optional
        Explicit bins in place of `r_min`, `r_max` and `n_bins`.
    """

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
        cuda: Optional[bool] = None,
        jitter=1e-3,
        shape=None,
        radial_bins=None,
        k_bins=None,
    ):
        if cuda is None:
            cuda = _cuda_available()
        self.cuda = bool(cuda)
        self.n0, self.k = int(n0), int(k)
        if graph is None:
            if points is None:
                raise TypeError("either `points` or `graph` is required")
            points = np.asarray(points)
            shape = points.shape[:-1] if shape is None else tuple(shape)
            graph = self.build_graph(points.reshape(-1, points.shape[-1]), n0=self.n0, k=self.k, cuda=self.cuda)
        elif shape is None:
            shape = (int(graph.points.shape[0]),) if points is None else np.shape(points)[:-1]
        if int(np.prod(shape)) != int(graph.points.shape[0]):
            raise ValueError(f"shape {shape} does not match the {graph.points.shape[0]} graph points")
        self.graph = graph
        pts = np.asarray(graph.points, dtype=np.float64)
        dim = int(pts.shape[1])
        if r_max is None and radial_bins is None:
            r_max = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) * (1.0 + 1e-3)
        if r_min is None and (radial_bins is None or k_bins is None):
            r_min = 0.5 * self._min_neighbour_distance(pts, np.asarray(graph.neighbors))
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

    @staticmethod
    def _min_neighbour_distance(pts, neighbors, chunk=2**18):
        n0 = pts.shape[0] - neighbors.shape[0]
        dmin = np.inf
        for start in range(0, neighbors.shape[0], chunk):
            sl = slice(start, min(start + chunk, neighbors.shape[0]))
            d = np.linalg.norm(pts[n0 + start : n0 + sl.stop, np.newaxis, :] - pts[neighbors[sl]], axis=-1)
            d = d[d > 0]
            if d.size:
                dmin = min(dmin, float(d.min()))
        if not np.isfinite(dmin):
            raise ValueError("could not infer `r_min` from the graph; pass it explicitly")
        return dmin

    @staticmethod
    def build_graph(points, *, n0: int, k: int, cuda: bool = False):
        """``graphgp.build_graph`` on ``points`` of shape ``(n, dim)`` with the
        dtypes of the CUDA kernels (float32 points, int32 neighbours)."""
        import graphgp as ggp

        points = np.asarray(points, dtype=np.float32).reshape(-1, np.shape(points)[-1])
        n0 = min(int(n0), points.shape[0])
        if cuda:
            graph = ggp.build_graph(jnp.asarray(points), n0=n0, k=int(k), cuda=True)
        else:
            with jax.enable_x64(False):
                graph = ggp.build_graph(jnp.asarray(points), n0=n0, k=int(k), cuda=False)
        indices = graph.indices
        if indices is not None:
            indices = indices.astype(jnp.int32)
        return ggp.Graph(
            graph.points.astype(jnp.float32), graph.neighbors.astype(jnp.int32), graph.offsets, indices
        )

    @property
    def points(self):
        """The points in their original order, shape ``shape + (dim,)``."""
        pts = np.asarray(self.graph.points)
        if self.graph.indices is not None:
            out = np.empty_like(pts)
            out[np.asarray(self.graph.indices)] = pts
            pts = out
        return pts.reshape(self.shape + (pts.shape[-1],))

    def apply_sqrt(self, xi, covariance):
        import graphgp as ggp

        rs = self.radial_bins

        def generate(x):
            return ggp.generate(self.graph, (rs, covariance), x, cuda=self.cuda)

        # NOTE, `lax.map` rather than `vmap` over the batch: the vjp of
        # graphgp's pure-JAX refinement (`fast_jit=True`) is wrong under
        # `vmap` (graphgp 0.0.3, 2026-10-27; the jvp is fine, `fast_jit=False`
        # is fine), and the CUDA primitives need no batching this way either.
        return jax.lax.map(generate, xi).reshape((xi.shape[0],) + self.shape)
