#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Philipp Frank

"""Gaussian process models with a tabulated isotropic kernel.

:class:`KernelGP` is the common base: a model whose excitations
``prefix + "xi"`` are standard normal and whose kernel is either a model
returning the covariance on radial bins (hierarchical: the kernel parameters
become part of the model's domain) or a fixed table (an array of covariance
values with ``radial_bins``, or a ``(radial_bins, cov_vals)`` tuple). The
generator :meth:`KernelGP.generate` maps excitations and a covariance table to
the field; :meth:`KernelGP.__call__` evaluates the kernel and generates.
:class:`GraphGP` generates on arbitrary points with graphgp;
:class:`nifty.re.multi_grid.ICRGP` on a multigrid with iterative charted
refinement. The correlated field composes these models with its amplitude
models.
"""

import dataclasses
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .model import Initializer, Model
from .tree_math import ShapeWithDtype, random_like

__all__ = ["KernelGP", "GraphGP", "parse_kernel"]


def parse_kernel(kernel, radial_bins=None):
    """Splits `kernel` into ``(model, radial_bins, cov_vals)``: a kernel model
    (with a ``radial_bins`` attribute unless `radial_bins` is given) and no
    fixed table, or no model and a fixed table from an array of covariance
    values on `radial_bins` or a ``(radial_bins, cov_vals)`` tuple."""
    if isinstance(kernel, Model):
        rs = getattr(kernel, "radial_bins", None) if radial_bins is None else radial_bins
        if rs is None:
            raise ValueError("a kernel model must have `radial_bins` or they must be given explicitly")
        return kernel, jnp.asarray(rs), None
    if isinstance(kernel, (tuple, list)) and len(kernel) == 2:
        if radial_bins is not None:
            raise ValueError("`radial_bins` given twice")
        rs, cov = kernel
        return None, jnp.asarray(rs), jnp.asarray(cov)
    if radial_bins is None:
        raise ValueError("a fixed kernel needs `radial_bins`: pass them or a `(radial_bins, cov_vals)` tuple")
    return None, jnp.asarray(radial_bins), jnp.asarray(kernel)


class KernelGP(Model):
    """Gaussian process with excitations ``prefix + "xi"`` of shape
    ``excitation_shape`` and a tabulated isotropic kernel.

    Parameters
    ----------
    kernel : Model, array or (radial_bins, cov_vals)
        Kernel model returning the covariance on its ``radial_bins`` (the
        model's domain joins the GP's domain), or a fixed covariance table.
    radial_bins : array, optional
        Radii of a fixed table given as an array (or of a kernel model without
        ``radial_bins``).
    shape, excitation_shape : tuples
        Shape of the field and of the excitations.
    prefix : str
        Name prefix of the excitations.
    """

    kernel: Optional[Model] = dataclasses.field(metadata=dict(static=False))
    cov_vals: Optional[jnp.ndarray] = dataclasses.field(metadata=dict(static=False))
    radial_bins: jnp.ndarray = dataclasses.field(metadata=dict(static=False))

    def __init__(self, kernel, *, radial_bins=None, shape, excitation_shape, prefix="", dtype=jnp.float64):
        self.kernel, self.radial_bins, self.cov_vals = parse_kernel(kernel, radial_bins)
        self.shape = tuple(int(s) for s in shape)
        self.excitation_shape = tuple(int(s) for s in excitation_shape)
        self.xi_name = str(prefix) + "xi"
        xi_dom = ShapeWithDtype(self.excitation_shape, dtype)
        domain = {self.xi_name: xi_dom}
        init = Initializer({self.xi_name: partial(random_like, primals=xi_dom)})
        if self.kernel is not None:
            domain = domain | self.kernel.domain
            init = init | self.kernel.init
        super().__init__(domain=domain, init=init)

    @property
    def hierarchical(self) -> bool:
        """Whether the kernel is a model (its parameters are inferred)."""
        return self.kernel is not None

    def covariance(self, p=None):
        """The covariance table on :attr:`radial_bins` for the parameters `p`."""
        return self.cov_vals if self.kernel is None else self.kernel(p)

    def generate(self, xi, cov_vals):
        """Field of shape ``batch + shape`` for excitations `xi` of shape
        ``batch + excitation_shape`` and the covariance table `cov_vals` on
        :attr:`radial_bins` (the linear map of the GP for a fixed table)."""
        raise NotImplementedError()

    def __call__(self, p):
        return self.generate(p[self.xi_name], self.covariance(p))


class GraphGP(KernelGP):
    """Gaussian process on arbitrary points, generated with graphgp (a
    nearest-neighbour Vecchia approximation on a k-d tree ordering). Requires
    the optional ``graphgp`` package (and ``graphgp_cuda`` for ``cuda=True``).

    Parameters
    ----------
    points : array ``(..., dim)`` or graphgp.Graph
        The points (the field gets the shape ``points.shape[:-1]``), or a graph
        built with :meth:`build_graph` (then the shape is ``(n_points,)``
        unless `shape` is given).
    kernel : Model, array or (radial_bins, cov_vals)
        See :class:`KernelGP`. graphgp holds the covariance constant beyond the
        last radial bin, so the bins must reach the largest separation.
    n0, k : int
        Points generated densely before the refinement; nearest predecessors
        each point is conditioned on.
    cuda : bool
        Use the graphgp CUDA extension (the default).
    shape : tuple, optional
        Shape of the field for a given graph.
    """

    graph: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(self, points, kernel, *, n0=512, k=16, cuda=True, shape=None, prefix="", radial_bins=None, dtype=jnp.float64):
        self.cuda = bool(cuda)
        self.n0, self.k = int(n0), int(k)
        if hasattr(points, "neighbors") and hasattr(points, "offsets"):
            graph = points
            shape = (int(graph.points.shape[0]),) if shape is None else tuple(shape)
        else:
            points = np.asarray(points)
            shape = points.shape[:-1] if shape is None else tuple(shape)
            graph = self.build_graph(points.reshape(-1, points.shape[-1]), n0=self.n0, k=self.k, cuda=self.cuda)
        if int(np.prod(shape)) != int(graph.points.shape[0]):
            raise ValueError(f"shape {shape} does not match the {graph.points.shape[0]} graph points")
        self.graph = graph
        super().__init__(kernel, radial_bins=radial_bins, shape=shape, excitation_shape=shape, prefix=prefix, dtype=dtype)

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
        return ggp.Graph(graph.points.astype(jnp.float32), graph.neighbors.astype(jnp.int32), graph.offsets, indices)

    @property
    def points(self):
        """The points in their original order, shape ``shape + (dim,)``."""
        pts = np.asarray(self.graph.points)
        if self.graph.indices is not None:
            out = np.empty_like(pts)
            out[np.asarray(self.graph.indices)] = pts
            pts = out
        return pts.reshape(self.shape + (pts.shape[-1],))

    def generate(self, xi, cov_vals):
        import graphgp as ggp

        rs = self.radial_bins
        batch = xi.shape[: xi.ndim - len(self.excitation_shape)]
        xi = xi.reshape((-1, int(np.prod(self.excitation_shape))))

        def gen(x):
            return ggp.generate(self.graph, (rs, cov_vals), x, cuda=self.cuda)

        # NOTE, `lax.map` rather than `vmap` over the batch: the vjp of graphgp's
        # pure-JAX refinement (`fast_jit=True`) is wrong under `vmap` (graphgp
        # 0.0.3, 2026-10-27; the jvp is fine), and the CUDA primitives need no
        # batching this way either.
        return jax.lax.map(gen, xi).reshape(batch + self.shape)
