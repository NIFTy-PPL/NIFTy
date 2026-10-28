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
:class:`ICRGP` on a multigrid with iterative charted
refinement. The correlated field composes these models with its amplitude
models.
"""

import dataclasses
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from .logger import logger
from .model import Initializer, Model
from .multi_grid.grid import Grid
from .multi_grid.kernel import ICRKernel
from .num import amend_unique_
from .tree_math import ShapeWithDtype, random_like

__all__ = ["KernelGP", "GraphGP", "ICRGP", "parse_kernel", "compress_indices", "refinement_matrices_cholesky"]


def parse_kernel(kernel, radial_bins=None, *, allow_callable=False):
    """Splits `kernel` into ``(model, radial_bins, cov_vals)``: a kernel model
    (with a ``radial_bins`` attribute unless `radial_bins` is given) and no
    fixed table, or no model and a fixed table from an array of covariance
    values on `radial_bins` or a ``(radial_bins, cov_vals)`` tuple.

    With `allow_callable`, a covariance *function* of two coordinate arrays
    ``k(x, y)`` (shapes ``(dim, ...)``, as the kernels of
    :class:`~nifty.re.multi_grid.ICRField`) is accepted as well, either
    directly (fixed) or as a model returning such a function (a model without
    ``radial_bins``); then ``radial_bins`` is ``None`` and `cov_vals` is the
    function (or ``None`` for the model).
    """
    if isinstance(kernel, Model):
        rs = getattr(kernel, "radial_bins", None) if radial_bins is None else radial_bins
        if rs is None:
            if allow_callable:
                return kernel, None, None
            raise ValueError("a kernel model must have `radial_bins` or they must be given explicitly")
        return kernel, jnp.asarray(rs), None
    if isinstance(kernel, (tuple, list)) and len(kernel) == 2:
        if radial_bins is not None:
            raise ValueError("`radial_bins` given twice")
        rs, cov = kernel
        return None, jnp.asarray(rs), jnp.asarray(cov)
    if callable(kernel):
        if not allow_callable:
            raise TypeError("a covariance function is not supported here; pass a table or a kernel model")
        if radial_bins is not None:
            raise ValueError("a covariance function has no `radial_bins`")
        return None, None, jax.tree_util.Partial(kernel)
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
    radial_bins: Optional[jnp.ndarray] = dataclasses.field(metadata=dict(static=False))

    def __init__(self, kernel, *, radial_bins=None, shape, excitation_shape, prefix="", dtype=jnp.float64, allow_callable=False):
        self.kernel, self.radial_bins, self.cov_vals = parse_kernel(kernel, radial_bins, allow_callable=allow_callable)
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

    @property
    def coordinate_kernel(self) -> bool:
        """Whether the kernel is a covariance function of coordinates rather
        than a table on radial bins."""
        return self.radial_bins is None

    def covariance(self, p=None):
        """The covariance table on :attr:`radial_bins` (or the covariance
        function for a coordinate kernel) for the parameters `p`."""
        return self.cov_vals if self.kernel is None else self.kernel(p)

    def generate(self, xi, cov_vals):
        """Field of shape ``batch + shape`` for excitations `xi` of shape
        ``batch + excitation_shape`` and the covariance table `cov_vals` on
        :attr:`radial_bins` (the linear map of the GP for a fixed table; a
        covariance function for a coordinate kernel)."""
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
    kernel : Model, array, (radial_bins, cov_vals) or callable
        See :class:`KernelGP`; additionally a covariance function ``k(x, y)``
        of two coordinate arrays (or a model returning one, such as
        :class:`~nifty.re.multi_grid.MaternCovarianceModel` in mode ``"ICR"``)
        is accepted and evaluated on the window coordinates. The compression
        of the windows groups them by their distance matrices, which is exact
        for isotropic kernels only; pass ``compress=None`` otherwise. graphgp holds the covariance constant beyond the
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


# ---- iterative charted refinement on a multigrid
# Gaussian process on a multigrid by iterative charted refinement (ICR).
#
# :class:`ICRGP` takes a multigrid (:class:`~nifty.re.multi_grid.Grid`), a
# kernel (model or fixed table, see :class:`~nifty.re.gaussian_process.KernelGP`)
# and the settings of the refinement: the window sizes, the compression of the
# refinement matrices and the chunk sizes. Given the covariance on the radial
# bins it generates the field level by level, each level in chunks of cells
# (``chunk_size`` cells at a time, recomputed in the backward pass), so that the
# memory of an application is bounded by the field and one chunk of gathered
# windows regardless of the grid's axes.
#
# The refinement matrices come from one Cholesky factorization per unique
# window covariance (``olf = L_fc L_cc^-1``, ``L_ff``); the unique windows are
# found by hashing the distance matrices quantized on the compression
# tolerance, followed by a greedy merge of the representatives.



def refinement_matrices_cholesky(cov, n_fsz: int):
    """Refinement matrices from one Cholesky factorization of the window
    covariance (coarse block first, fine block last).

    With ``L = [[L_cc, 0], [L_fc, L_ff]]`` and ``L L^T = cov``:
    ``olf = C_fc C_cc^-1 = L_fc L_cc^-1`` and
    ``L_ff L_ff^T = C_ff - C_fc C_cc^-1 C_cf``. The fine excitations are mapped
    through the triangular ``L_ff``.
    """
    n_csz = cov.shape[0] - n_fsz
    L = jnp.linalg.cholesky(cov)
    L_cc, L_fc, L_ff = L[:n_csz, :n_csz], L[n_csz:, :n_csz], L[n_csz:, n_csz:]
    olf = jax.scipy.linalg.solve_triangular(L_cc.T, L_fc.T, lower=False).T
    return olf, L_ff


def _quantized_hashes(dist, atol, rtol, keys):
    """Two 64-bit hashes per distance matrix, quantized on the tolerance grid
    (entries within ``atol + rtol * |b|`` fall into the same or neighboring
    bins of ``log(d + atol / rtol)`` with width ``log(1 + rtol)``)."""
    eps = atol / rtol
    q = jnp.floor(jnp.log(dist + eps) / jnp.log1p(rtol)).astype(jnp.int64)
    q = q.reshape(q.shape[0], -1).astype(jnp.uint64)
    return jnp.stack([(q * k[jnp.newaxis, :]).sum(axis=1) for k in keys], axis=1)


def _compress_scan(kernel, cells, level, rtol, atol):
    """Greedy tolerance merge of the distance matrices of `cells`."""

    def get_matrix(idx):
        return kernel.get_kernel_distmat(idx, level)[0]

    def step(u, i):
        u, invid = amend_unique_(u, get_matrix(cells[:, i]), axis=0, atol=atol, rtol=rtol)
        return u, invid

    shp = jax.eval_shape(get_matrix, cells[:, 0]).shape
    # the merge of n representatives yields at most n uniques
    unique = jnp.full((cells.shape[1] + 1,) + shp, jnp.nan)
    unique, invid = jax.lax.scan(step, unique, np.arange(cells.shape[1]))
    _, idx, inverse = np.unique(np.asarray(invid), return_index=True, return_inverse=True)
    assert np.all(np.isnan(unique[idx.size :]))
    return cells[:, idx], inverse.reshape(-1)


def compress_indices(
    kernel: ICRKernel,
    cells,
    level: int,
    *,
    rtol=1e-5,
    atol=1e-10,
    buffer_size=10_000,
    chunk_size=32768,
):
    """Groups the `cells` (shape ``(ndim, n)``) of `level` by their window
    distance matrices: returns the representatives ``(ndim, n_unique)`` and
    the index of each cell's representative ``(n,)``.

    The distance matrices are quantized on the tolerance grid and hashed in
    chunks, identical hashes are grouped, and, if there are at most
    `buffer_size` groups, the group representatives are merged with a greedy
    tolerance scan (quadratic in their number) so that groups split by a bin
    boundary end up together. Every cell is mapped to a representative whose
    distance matrix is within (twice) the tolerance of its own.
    """
    if atol <= 0.0 or rtol <= 0.0:
        raise ValueError("compression needs `atol > 0` and `rtol > 0`")
    cells = np.asarray(cells)
    n = cells.shape[1]
    n_entries = int(
        np.prod(
            jax.eval_shape(
                lambda i: kernel.get_kernel_distmat(i, level)[0], jnp.asarray(cells[:, :1])
            ).shape[1:]
        )
    )
    rng = np.random.default_rng(0)
    keys = rng.integers(1, 2**63 - 1, size=(2, n_entries), dtype=np.int64)
    keys = jnp.asarray(keys.astype(np.uint64) | np.uint64(1))

    @jax.jit
    def hashes_of(idx):
        return _quantized_hashes(kernel.get_kernel_distmat(idx, level)[0], atol, rtol, keys)

    hashes = np.empty((n, 2), dtype=np.uint64)
    for start in range(0, n, chunk_size):
        sl = slice(start, min(start + chunk_size, n))
        hashes[sl] = np.asarray(hashes_of(jnp.asarray(cells[:, sl])))
    _, first, inverse = np.unique(hashes, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    uids = cells[:, first]
    if 1 < uids.shape[1] <= buffer_size:
        uids, inverse2 = _compress_scan(kernel, jnp.asarray(uids), level, rtol, atol)
        inverse = inverse2[inverse]
        uids = np.asarray(uids)
    elif uids.shape[1] > buffer_size:
        logger.info(
            f"{uids.shape[1]} hash groups exceed `buffer_size` {buffer_size};"
            " the representatives are not merged"
        )
    return uids, inverse


@partial(jax.tree_util.register_dataclass, data_fields=["invids", "ucoords"], meta_fields=["index_shape", "index_lo", "splits", "fine_shape", "n_f"])
@dataclass(frozen=True)
class _Level:
    index_shape: Tuple[int, ...]  # shape of the block of refined indices
    index_lo: Tuple[int, ...]  # first refined index along each axis
    splits: Tuple[int, ...]
    fine_shape: Tuple[int, ...]
    n_f: int
    invids: jnp.ndarray  # (n_cells,) index of the unique matrix of each cell
    ucoords: jnp.ndarray  # (n_unique, dim, n) coordinates of the unique windows' points

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.index_shape))

    @property
    def n_unique(self) -> int:
        return int(self.ucoords.shape[0])


def _window_coordinates(kernel, uids, level):
    """Coordinates ``(n_unique, dim, n_coarse + n_fine)`` of the points of the
    windows refining the cells `uids` (``(ndim, n_unique)``)."""
    _, ((idc, _), (idf, _)) = kernel.get_output_input_indices(uids, level)
    gc = kernel.grid.at(level).index2coord(idc)
    gf = kernel.grid.at(level + 1).index2coord(idf)
    return jnp.moveaxis(jnp.concatenate((gc, gf), axis=-1), 0, 1)


def _distances(coords):
    """Pairwise distances ``(..., n, n)`` of points ``(..., dim, n)``."""
    d = coords[..., :, :, jnp.newaxis] - coords[..., :, jnp.newaxis, :]
    return jnp.sqrt(jnp.sum(d * d, axis=-3))


class ICRGP(KernelGP):
    """Gaussian process on a multigrid, generated by iterative charted
    refinement; the field lives on the finest level, the excitations are one
    flat array holding the levels one after the other (coarsest first, each
    level in the C order of its grid shape; see :meth:`split_excitations`).

    Parameters
    ----------
    grid : nifty.re.multi_grid.Grid
        The multigrid.
    kernel : Model, array or (radial_bins, cov_vals)
        See :class:`~nifty.re.gaussian_process.KernelGP`. The covariance is
        interpolated linearly between the radial bins and is zero beyond the
        last one, so the bins must reach the largest distance in any
        refinement window (a warning is logged otherwise); too few bins per
        decade make the interpolated window covariances indefinite and the
        Cholesky factorization fails with NaNs (for a steep spectrum 35 bins
        per decade were indefinite and 135 fine; keep at least 100 per decade).
    window_size : tuple or None
        Coarse window of the refinement at each level (grid default: 3 along
        every axis).
    compress : dict or None
        Tolerances of the window compression (``rtol``, ``atol``,
        ``buffer_size``, ``chunk_size``), see :func:`compress_indices`;
        ``None`` keeps every window.
    chunk_size : int
        Number of cells refined at once.
    kernel_chunk : int
        Number of unique windows whose refinement matrices are built at once
        (the distances are recomputed from the window coordinates in the
        backward pass, so the memory of the kernel build is that of one
        chunk).
    cholesky_jitter : float
        Added to the diagonal of every window covariance relative to its mean
        before the Cholesky factorization (windows at a grid boundary contain
        coincident points).
    prefix : str
        Name prefix of the excitations ``prefix + "xi"``.
    radial_bins : array, optional
        Radii of a fixed table given as an array.
    """

    _base_dist: jnp.ndarray = field(metadata=dict(static=False))
    _base_coords: jnp.ndarray = field(metadata=dict(static=False))
    _levels: Tuple[_Level, ...] = field(metadata=dict(static=False))

    def __init__(
        self,
        grid: Grid,
        kernel,
        *,
        window_size=None,
        compress: Optional[dict] = dict(rtol=1e-5, atol=1e-10, buffer_size=10_000),
        chunk_size=2**16,
        kernel_chunk=4096,
        cholesky_jitter=1e-6,
        prefix="",
        radial_bins=None,
        dtype=jnp.float64,
    ):
        self.multigrid = grid
        self.icr_kernel = ICRKernel(grid, None, window_size=window_size)
        self.chunk_size = int(chunk_size)
        self.kernel_chunk = int(kernel_chunk)
        self.cholesky_jitter = float(cholesky_jitter)
        self._compress = None if not compress else tuple(sorted(dict(compress).items()))

        # the index arithmetic of the grids is slow op by op; always run it under jit
        self._base_dist = jax.jit(lambda i: self.icr_kernel.get_kernel_distmat(i, None))(jnp.array([-1]))
        _, ((base_ids, _),) = self.icr_kernel.get_output_input_indices(jnp.array([-1]), None)
        self._base_coords = jax.jit(grid.at(0).index2coord)(jnp.asarray(base_ids))  # (dim, n0)
        levels = []
        for lvl in range(grid.depth):
            index = np.asarray(grid.at(lvl).refined_indices())
            index_shape = tuple(int(s) for s in index.shape[1:])
            cells = index.reshape(index.shape[0], -1)
            if self.compress is not None:
                uids, invids = compress_indices(self.icr_kernel, cells, lvl, **self.compress)
            else:
                uids, invids = cells, np.arange(cells.shape[1])
            ucoords = jax.jit(lambda i, lvl=lvl: _window_coordinates(self.icr_kernel, i, lvl))(jnp.asarray(uids))
            n_f = int(np.prod(grid.at(lvl).splits))
            logger.info(f"ICRGP level {lvl}: {cells.shape[1]} cells, {uids.shape[1]} unique windows")
            levels.append(
                _Level(
                    index_shape=index_shape,
                    index_lo=tuple(int(i) for i in index[(slice(None),) + (0,) * len(index_shape)]),
                    splits=tuple(int(s) for s in grid.at(lvl).splits),
                    fine_shape=tuple(int(s) for s in grid.at(lvl + 1).shape),
                    n_f=int(n_f),
                    invids=jnp.asarray(invids, dtype=jnp.int32),
                    ucoords=jnp.asarray(ucoords),
                )
            )
        self._levels = tuple(levels)
        self._level_shapes = tuple(tuple(int(s) for s in grid.at(l).shape) for l in range(grid.depth + 1))
        sizes = [int(np.prod(s)) for s in self._level_shapes]
        self._level_offsets = tuple(int(o) for o in np.cumsum([0] + sizes))

        super().__init__(
            kernel, radial_bins=radial_bins, shape=self._level_shapes[-1], excitation_shape=(sum(sizes),), prefix=prefix,
            dtype=dtype, allow_callable=True,
        )
        d_max = float(np.max(self._base_dist))
        for l in levels:
            for start in range(0, l.n_unique, self.kernel_chunk):
                d_max = max(d_max, float(np.max(jax.jit(_distances)(l.ucoords[start : start + self.kernel_chunk]))))
        self.max_distance = d_max
        if self.radial_bins is not None and d_max > float(self.radial_bins[-1]):
            logger.warning(
                f"the covariance table ends at {float(self.radial_bins[-1]):.4g} but the refinement windows reach"
                f" distance {d_max:.4g}; the covariance is zero beyond the table"
            )

    @property
    def compress(self) -> Optional[dict]:
        """Settings of the window compression (``None``: no compression)."""
        return None if self._compress is None else dict(self._compress)

    @property
    def depth(self) -> int:
        return self.multigrid.depth

    @property
    def window_size(self):
        return self.icr_kernel.window_size

    @property
    def n_unique(self) -> Tuple[int, ...]:
        """Number of unique refinement windows per level."""
        return tuple(l.n_unique for l in self._levels)

    def _jittered(self, cov):
        eye = jnp.eye(cov.shape[-1], dtype=cov.dtype)
        diag = jnp.mean(jnp.diagonal(cov, axis1=-2, axis2=-1), axis=-1)
        return cov + self.cholesky_jitter * diag[..., jnp.newaxis, jnp.newaxis] * eye

    def matrices(self, covariance):
        """Base Cholesky factor and the per-level ``(olf, L_ff)`` of the
        unique windows for the `covariance` on :attr:`radial_bins` (a table,
        interpolated linearly at the window distances) or, for a coordinate
        kernel, the covariance function ``k(x, y)`` evaluated on the window
        coordinates."""
        if self.coordinate_kernel:
            k = covariance
            k = vmap(k, in_axes=(None, -1), out_axes=-1)
            k = vmap(k, in_axes=(-1, None), out_axes=-1)

            def window_cov(c):  # c: (..., dim, n) -> (..., n, n)
                return jnp.vectorize(lambda cc: k(cc, cc), signature="(d,n)->(n,n)")(c)

            base = jnp.linalg.cholesky(self._jittered(window_cov(self._base_coords)))
        else:
            rs = self.radial_bins

            def kf(d):
                return jnp.interp(d, rs, covariance, right=0.0)

            def window_cov(c):
                return kf(_distances(c))

            base = jnp.linalg.cholesky(self._jittered(kf(self._base_dist)))
        mats = []
        for l in self._levels:
            chunk = min(self.kernel_chunk, l.n_unique)
            n_chunks = -(-l.n_unique // chunk)
            pad = n_chunks * chunk - l.n_unique
            coords = jnp.concatenate((l.ucoords, jnp.repeat(l.ucoords[-1:], pad, axis=0)))
            coords = coords.reshape((n_chunks, chunk) + l.ucoords.shape[1:])

            @jax.checkpoint
            def build(c, n_f=l.n_f):
                # keep XLA from folding the distance tables of all windows into constants
                c = jax.lax.optimization_barrier(c)
                return vmap(partial(refinement_matrices_cholesky, n_fsz=n_f))(self._jittered(window_cov(c)))

            olf, lff = jax.lax.map(build, coords)
            mats.append((olf.reshape((-1,) + olf.shape[2:])[: l.n_unique], lff.reshape((-1,) + lff.shape[2:])[: l.n_unique]))
        return base, tuple(mats)

    def split_excitations(self, xi):
        """The per-level arrays of a flat vector of excitations."""
        return [
            xi[..., o : o + int(np.prod(s))].reshape(xi.shape[:-1] + s)
            for o, s in zip(self._level_offsets, self._level_shapes)
        ]

    def _refine(self, lvl, coarse, xi_fine, olf, lff):
        L = self._levels[lvl]
        n = L.n_cells
        chunk = min(self.chunk_size, n)
        n_chunks = -(-n // chunk)
        pad = n_chunks * chunk - n
        kid = jnp.concatenate((L.invids, jnp.repeat(L.invids[-1:], pad))).reshape(n_chunks, chunk)
        starts = jnp.arange(n_chunks) * chunk
        index_lo = jnp.asarray(L.index_lo)

        @jax.checkpoint
        def body(start, kid):
            flat = jnp.minimum(start + jnp.arange(chunk), n - 1)
            cells = jnp.stack(jnp.unravel_index(flat, L.index_shape)) + index_lo[:, jnp.newaxis]
            _, ((gc, _), (gf, _)) = self.icr_kernel.get_output_input_indices(cells, lvl)
            xc = coarse[tuple(gc)]
            xf = xi_fine[tuple(gf)]
            return jnp.einsum("cfw,cw->cf", olf[kid], xc) + jnp.einsum("cfg,cg->cf", lff[kid], xf)

        res = jax.lax.map(lambda xs: body(*xs), (starts, kid))
        res = res.reshape(-1, L.n_f)[:n]
        nd = len(L.index_shape)
        res = res.reshape(L.index_shape + L.splits)
        order = sum(((i, nd + i) for i in range(nd)), ())
        return res.transpose(order).reshape(L.fine_shape)

    def generate(self, xi, cov_vals):
        base, mats = self.matrices(cov_vals)
        batch = xi.shape[:-1]
        xi = xi.reshape((-1, xi.shape[-1]))

        def single(x):
            xs = self.split_excitations(x)
            f = (base @ xs[0].ravel()).reshape(self._level_shapes[0])
            for lvl, (olf, lff) in enumerate(mats):
                f = self._refine(lvl, f, xs[lvl + 1], olf, lff)
            return f

        return vmap(single)(xi).reshape(batch + self.shape)
