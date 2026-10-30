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
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .logger import logger
from .model import Initializer, Model
from .multi_grid.grid import Grid
from .multi_grid.kernel import ICRKernel
from .num import amend_unique_
from .tree_math import ShapeWithDtype, random_like

__all__ = ["KernelGP", "GraphGP", "ICRGP", "ICRStructure", "build_icr_structure", "parse_kernel", "compress_indices", "refinement_matrices_cholesky"]


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
    """Gaussian process with standard-normal excitations ``prefix + "xi"`` and
    a tabulated isotropic kernel.

    The excitations are one array of shape ``excitation_shapes[0]`` or, for a
    multigrid generator, a list of arrays (one per level, shapes
    ``excitation_shapes``); ``structured`` tells which.

    Parameters
    ----------
    kernel : Model, array or (radial_bins, cov_vals)
        Kernel model returning the covariance on its ``radial_bins`` (the
        model's domain joins the GP's domain), or a fixed covariance table.
    radial_bins : array, optional
        Radii of a fixed table given as an array (or of a kernel model without
        ``radial_bins``).
    shape : tuple
        Shape of the field.
    excitation_shapes : tuple of tuples
        Shapes of the excitation arrays (one entry: a single array).
    prefix : str
        Name prefix of the excitations.
    """

    kernel: Optional[Model] = dataclasses.field(metadata=dict(static=False))
    cov_vals: Optional[jnp.ndarray] = dataclasses.field(metadata=dict(static=False))
    radial_bins: Optional[jnp.ndarray] = dataclasses.field(metadata=dict(static=False))

    def __init__(self, kernel, *, radial_bins=None, shape, excitation_shapes, prefix="", dtype=jnp.float64, allow_callable=False):
        self.kernel, self.radial_bins, self.cov_vals = parse_kernel(kernel, radial_bins, allow_callable=allow_callable)
        self.shape = tuple(int(s) for s in shape)
        self.excitation_shapes = tuple(tuple(int(s) for s in sh) for sh in excitation_shapes)
        self.xi_name = str(prefix) + "xi"
        xi_dom = [ShapeWithDtype(sh, dtype) for sh in self.excitation_shapes]
        xi_dom = xi_dom[0] if len(xi_dom) == 1 else xi_dom
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
    def structured(self) -> bool:
        """Whether the excitations are a list of arrays rather than one array."""
        return len(self.excitation_shapes) > 1

    @property
    def excitation_shape(self):
        """The shape of the (single) excitation array."""
        if self.structured:
            raise AttributeError("the excitations are a list of arrays, see `excitation_shapes`")
        return self.excitation_shapes[0]

    def batch_shape(self, xi):
        """Leading batch dimensions of excitations `xi` (an array or a list of arrays)."""
        x = xi[0] if self.structured else xi
        return tuple(x.shape[: x.ndim - len(self.excitation_shapes[0])])

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
        """Field of shape ``batch + shape`` for excitations `xi` (an array of
        shape ``batch + excitation_shapes[0]``, or a list of such arrays for a
        structured generator) and the covariance table `cov_vals` on
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
    points : array ``(..., dim)``, graphgp.Graph or ShardedGraph
        The points (the field gets the shape ``points.shape[:-1]``), or a graph
        built with :meth:`build_graph` (then the shape is ``(n_points,)``
        unless `shape` is given). A ``jax.Array`` sharded along its leading
        axis implies ``mesh`` (its mesh).
    kernel : Model, array, (radial_bins, cov_vals) or callable
        See :class:`KernelGP`; graphgp holds the covariance constant beyond the
        last radial bin, so the bins must reach the largest separation.
    n0, k : int
        Points generated densely before the refinement; nearest predecessors
        each point is conditioned on.
    cuda : bool
        Use the graphgp CUDA extension (the default).
    shape : tuple, optional
        Shape of the field for a given graph.
    mesh : jax.sharding.Mesh, optional
        One-dimensional device mesh: the field (and its excitations) is sharded
        along its leading axis in contiguous blocks, generated with compacted
        levels and a halo exchange (:mod:`nifty.re.graphgp_sharded`). For
        points the graph is built on the devices (:func:`~nifty.re.graphgp_sharded.build_sharded_graph`);
        a global ``graphgp.Graph`` is converted (:func:`~nifty.re.graphgp_sharded.sharded_graph_from_graph`).
        The leading axis must be divisible by the number of devices.
    """

    graph: Any = dataclasses.field(metadata=dict(static=False))
    mesh: Optional[Mesh] = dataclasses.field(metadata=dict(static=True))

    def __init__(self, points, kernel, *, n0=512, k=16, cuda=True, shape=None, prefix="", radial_bins=None, dtype=jnp.float64, mesh=None):
        from .graphgp_sharded import ShardedGraph

        self.cuda = bool(cuda)
        self.n0, self.k = int(n0), int(k)
        if mesh is None and isinstance(points, jax.Array) and isinstance(points.sharding, NamedSharding) and points.sharding.spec:
            if points.sharding.spec[0] is not None:
                mesh = points.sharding.mesh
            elif any(s is not None for s in points.sharding.spec):
                logger.warning(
                    f"GraphGP: the points are sharded along a non-leading axis ({points.sharding.spec}); only the leading axis can be sharded, the field is generated unsharded"
                )
        if mesh is not None and len(mesh.axis_names) != 1:
            raise ValueError(f"mesh must be one-dimensional (one sharded axis per field), got axes {mesh.axis_names}")
        self.mesh = mesh
        if isinstance(points, ShardedGraph):
            graph = points
            if mesh is None:
                raise TypeError("a ShardedGraph needs the mesh it was built for")
            n_points = graph.n_shards * graph.block_size
            shape = (n_points,) if shape is None else tuple(shape)
        elif hasattr(points, "neighbors") and hasattr(points, "offsets"):
            graph = points
            n_points = int(graph.points.shape[0])
            shape = (n_points,) if shape is None else tuple(shape)
            if mesh is not None:
                graph = self.build_graph(graph, n0=self.n0, k=self.k, cuda=self.cuda, mesh=mesh)
        else:
            shape = tuple(points.shape[:-1]) if shape is None else tuple(shape)
            graph = self.build_graph(points, n0=self.n0, k=self.k, cuda=self.cuda, mesh=mesh)
            n_points = graph.n_shards * graph.block_size if mesh is not None else int(graph.points.shape[0])
        if int(np.prod(shape)) != n_points:
            raise ValueError(f"shape {shape} does not match the {n_points} graph points")
        if mesh is not None:
            S = mesh.shape[mesh.axis_names[0]]
            if graph.n_shards != S:
                raise ValueError(f"graph sharded {graph.n_shards}-fold, mesh has {S} devices")
            if shape[0] % S:
                raise ValueError(f"leading axis {shape[0]} is not divisible into {S} blocks")
            put = partial(jax.device_put, device=NamedSharding(mesh, P(mesh.axis_names[0])))
            rep = partial(jax.device_put, device=NamedSharding(mesh, P()))
            graph = jax.tree_util.tree_map(put, graph)
            graph = dataclasses.replace(graph, init_src=rep(graph.init_src), init_pos=rep(graph.init_pos), initial_points=rep(graph.initial_points))
        self.graph = graph
        super().__init__(kernel, radial_bins=radial_bins, shape=shape, excitation_shapes=(shape,), prefix=prefix, dtype=dtype)

    @staticmethod
    def build_graph(points, *, n0: int, k: int, cuda: bool = False, mesh=None):
        """``graphgp.build_graph`` on ``points`` of shape ``(..., dim)`` with the
        dtypes of the CUDA kernels (float32 points, int32 neighbours). With a
        `mesh` the graph is built on the devices and returned as a
        :class:`~nifty.re.graphgp_sharded.ShardedGraph` (``points`` may also be
        a global ``graphgp.Graph`` to convert)."""
        import graphgp as ggp

        if mesh is not None:
            from .graphgp_sharded import build_sharded_graph, sharded_graph_from_graph

            if hasattr(points, "neighbors") and hasattr(points, "offsets"):
                return sharded_graph_from_graph(points, mesh.shape[mesh.axis_names[0]])
            return build_sharded_graph(points, mesh, n0=n0, k=k, cuda=cuda)
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
        if self.mesh is not None:
            g = self.graph
            pts = np.asarray(g.ext_points)[:, : g.block_size].reshape(-1, np.shape(g.ext_points)[-1])
            return pts.reshape(self.shape + (pts.shape[-1],))
        pts = np.asarray(self.graph.points)
        if self.graph.indices is not None:
            out = np.empty_like(pts)
            out[np.asarray(self.graph.indices)] = pts
            pts = out
        return pts.reshape(self.shape + (pts.shape[-1],))

    # ---- sharding (mesh given)
    @property
    def axis(self):
        return None if self.mesh is None else self.mesh.axis_names[0]

    @property
    def sharding(self):
        """Sharding of the excitations and of the field (leading axis on the mesh)."""
        if self.mesh is None:
            return None
        return NamedSharding(self.mesh, P(self.axis, *([None] * (len(self.shape) - 1))))

    @property
    def shardings(self):
        """Per excitation block (one): its sharding."""
        return (self.sharding,)

    def shard(self, x):
        """Places the excitations of a parameter dict on the mesh."""
        x = dict(x)
        x[self.xi_name] = jax.device_put(np.asarray(x[self.xi_name]), self.sharding)
        return x

    def generate(self, xi, cov_vals):
        import graphgp as ggp

        rs = self.radial_bins
        batch = self.batch_shape(xi)
        xi = xi.reshape((-1, int(np.prod(self.excitation_shape))))
        if self.mesh is not None:
            from .graphgp_sharded import _ShardedGenerate

            gen = _ShardedGenerate(self.graph, self.mesh, cuda=self.cuda, radial_bins=rs)
            if len(batch) == 0:
                return gen(xi[0], cov_vals).reshape(self.shape)
            return jax.vmap(lambda x: gen(x, cov_vals))(xi).reshape(batch + self.shape)

        def gen(x):
            return ggp.generate(self.graph, (rs, cov_vals), x, cuda=self.cuda)

        # NOTE, requires graphgp with the fix of the fast_jit vjp under vmap
        # (branch fix-fast-jit-vmap, 2026-10-28); earlier versions need lax.map here
        return jax.vmap(gen)(xi).reshape(batch + self.shape)


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


def _compress_scan(kernel, cells, level, rtol, atol, buffer_size):
    """Greedy tolerance merge of the distance matrices of `cells`: every matrix
    is compared with the uniques found so far (at most `buffer_size` of them;
    a ``ValueError`` if more are needed) and joins the first one within
    ``atol + rtol * |.|``, as dust's ``compress_kernel(method="scan")``."""

    def get_matrix(idx):
        return kernel.get_kernel_distmat(idx, level)[0]

    def step(u, i):
        u, invid = amend_unique_(u, get_matrix(cells[:, i]), axis=0, atol=atol, rtol=rtol)
        return u, invid

    shp = jax.eval_shape(get_matrix, cells[:, 0]).shape
    n_buf = min(int(buffer_size), cells.shape[1]) + 1  # the merge of n representatives yields at most n uniques
    unique = jnp.full((n_buf,) + shp, jnp.nan)
    unique, invid = jax.lax.scan(step, unique, np.arange(cells.shape[1]))
    _, idx, inverse = np.unique(np.asarray(invid), return_index=True, return_inverse=True)
    if idx.size >= n_buf or not np.all(np.isnan(unique[idx.size :])):
        raise ValueError(f"more than `buffer_size` {buffer_size} unique windows after the merge at level {level}")
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
    chunks, identical hashes are grouped, and the group representatives are
    merged with a greedy tolerance scan (linear in their number, times the
    number of uniques) so that groups split by a bin boundary end up together;
    `buffer_size` bounds the number of uniques after the merge (``ValueError``
    beyond it). Every cell is mapped to a representative whose distance matrix
    is within (twice) the tolerance of its own.
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
    if uids.shape[1] > 1:
        # always merge the representatives (dust's behaviour): the quantization splits one tolerance
        # ball into many hash groups at fine levels (1.0 M groups for 4263 uniques at nside 128 x 516
        # bins, 2026-10-29); `buffer_size` caps the uniques after the merge, not the groups
        if uids.shape[1] > buffer_size:
            logger.info(f"level {level}: merging {uids.shape[1]} hash groups (more than `buffer_size` {buffer_size}) with the tolerance scan")
        # the scan compares every representative with the whole uniques buffer: start small and
        # grow it (doubling) when the uniques exceed it, up to `buffer_size`
        n_buf = min(4096, buffer_size)
        while True:
            try:
                uids_m, inverse2 = _compress_scan(kernel, jnp.asarray(uids), level, rtol, atol, n_buf)
                break
            except ValueError:
                if n_buf >= buffer_size:
                    raise
                n_buf = min(2 * n_buf, buffer_size)
        inverse = inverse2[inverse]
        uids = np.asarray(uids_m)
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


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["halo_ids", "exp", "imp_src", "imp_pos", "imp_slot", "pos_of_global"],
    meta_fields=["sharded", "coarse_full", "block_cells", "block_coarse", "block_fine", "halo_size", "block_index_shape", "block_fine_shape"],
)
@dataclass(frozen=True)
class _ShardLevel:
    """Static per-shard tables of one refinement step (level l -> l + 1) on a mesh.

    ``sharded``: the fine level is sharded along the first axis (else it is
    computed replicated); ``coarse_full``: the coarse level is replicated, so
    windows gather with global indices; otherwise each shard holds its coarse
    block extended by a halo ``[block | halo | junk]`` and the tables say
    which halo values come from where: ``halo_ids`` (S, H) sorted global flat
    indices of the halo cells (padded with the level size), ``exp`` (S, E)
    local positions of the own values other shards need, ``imp_src`` /
    ``imp_pos`` / ``imp_slot`` (S, I) where the gathered exports go.
    """

    sharded: bool
    coarse_full: bool
    block_cells: int
    block_coarse: int
    block_fine: int
    halo_size: int
    block_index_shape: Tuple[int, ...]
    block_fine_shape: Tuple[int, ...]
    halo_ids: Optional[jnp.ndarray]
    exp: Optional[jnp.ndarray]
    imp_src: Optional[jnp.ndarray]
    imp_pos: Optional[jnp.ndarray]
    imp_slot: Optional[jnp.ndarray]
    pos_of_global: Optional[jnp.ndarray] = None  # (S, coarse size): position of every global coarse cell in the shard's extended array (junk if absent)


def _icr_shard_tables(icr_kernel, levels_in, level_shapes, mesh, chunk_size):
    """Per-shard halo tables of every refinement step for `mesh` (see :class:`_ShardLevel`)."""
    ax = mesh.axis_names[0]
    S = int(mesh.shape[ax])
    shapes = level_shapes
    sizes = [int(np.prod(sh)) for sh in shapes]
    for l, size in enumerate(sizes):
        if size % S:
            raise ValueError(f"level {l} of size {size} is not divisible into {S} shards")
    sharded = [sh[0] % S == 0 and sh[0] >= S for sh in shapes]
    sharded[0] = False  # the base level is generated replicated (dense, from the gathered excitations)
    first = next((i for i, v in enumerate(sharded) if v), len(sharded))
    if any(sharded[:first]) or not all(sharded[first:]):
        raise ValueError("the sharded levels must be the finer ones (replicated coarse levels first)")
    n_rep = sum(sizes[:first])
    msg = (
        f"ICRGP on {S} shards: levels 0-{first - 1} replicated ({', '.join(str(sh) for sh in shapes[:first])}; "
        f"{n_rep} of {sizes[-1]} finest-level pixels), levels {first}-{len(shapes) - 1} sharded along the first axis"
    )
    if first == len(shapes):
        msg = f"ICRGP on {S} shards: no level is sharded (the first axis of every level is not divisible by {S}); everything is replicated"
    if first > 1 and n_rep > 0.05 * sizes[-1] or first == len(shapes):
        logger.warning(msg + " -- choose a mesh size that divides the first axis of the coarse levels, or a finer base grid")
    else:
        logger.info(msg)
    levels = []
    for lvl, L in enumerate(levels_in):
        fine_sharded = sharded[lvl + 1]
        coarse_full = not sharded[lvl]
        if fine_sharded and (L.index_lo[0] != 0 or L.index_shape[0] != shapes[lvl][0]):
            raise ValueError("sharding needs a first grid axis without padding (e.g. HEALPix)")
        block_cells = L.n_cells // S if fine_sharded else L.n_cells
        block_fine = sizes[lvl + 1] // S if fine_sharded else sizes[lvl + 1]
        block_coarse = sizes[lvl] // S if not coarse_full else sizes[lvl]
        bis = ((L.index_shape[0] // S,) + L.index_shape[1:]) if fine_sharded else L.index_shape
        bfs = ((L.fine_shape[0] // S,) + L.fine_shape[1:]) if fine_sharded else L.fine_shape
        if not fine_sharded or coarse_full:
            levels.append(_ShardLevel(fine_sharded, coarse_full, block_cells, block_coarse, block_fine, 0, bis, bfs, None, None, None, None, None))
            continue
        # halo of every shard: foreign coarse cells in the windows of its block
        cshape = shapes[lvl]
        win = jax.jit(lambda c, lvl=lvl: icr_kernel.get_output_input_indices(c, lvl)[1][0][0])
        halos = []
        for sidx in range(S):
            foreign = []
            for start in range(0, block_cells, chunk_size):
                flat = np.arange(sidx * block_cells + start, sidx * block_cells + min(start + chunk_size, block_cells))
                cells = np.stack(np.unravel_index(flat, L.index_shape)) + np.asarray(L.index_lo)[:, None]
                gc = np.asarray(win(jnp.asarray(cells)))  # (ndim, n, nw)
                gidx = np.ravel_multi_index(tuple(gc.reshape(gc.shape[0], -1)), cshape)
                foreign.append(np.unique(gidx[(gidx < sidx * block_coarse) | (gidx >= (sidx + 1) * block_coarse)]))
            halos.append(np.unique(np.concatenate(foreign)) if foreign else np.zeros(0, np.int64))
        H = max(len(h) for h in halos)
        needed = [[] for _ in range(S)]  # needed[owner]: global ids other shards import
        for h in halos:
            for g in h:
                needed[int(g) // block_coarse].append(int(g))
        exp_lists = [np.unique(np.array(needed[t], dtype=np.int64)) for t in range(S)]
        E = max((len(e) for e in exp_lists), default=0)
        exp_pos = {int(g): i for t in range(S) for i, g in enumerate(exp_lists[t])}
        I = H
        junk = block_coarse + H
        halo_ids = np.full((S, H), sizes[lvl], dtype=np.int64)
        exp = np.zeros((S, max(E, 1)), dtype=np.int64)
        imp_src = np.zeros((S, max(I, 1)), dtype=np.int64)
        imp_pos = np.zeros((S, max(I, 1)), dtype=np.int64)
        imp_slot = np.full((S, max(I, 1)), junk, dtype=np.int64)
        for sidx in range(S):
            halo_ids[sidx, : len(halos[sidx])] = halos[sidx]
            exp[sidx, : len(exp_lists[sidx])] = exp_lists[sidx] - sidx * block_coarse
            for j, g in enumerate(halos[sidx]):
                imp_src[sidx, j] = int(g) // block_coarse
                imp_pos[sidx, j] = exp_pos[int(g)]
                imp_slot[sidx, j] = block_coarse + j
        logger.info(f"ICRGP level {lvl + 1} on {S} shards: block {block_cells} cells, halo up to {H} coarse cells, exports up to {E}")
        # the window indices are translated with one gather into this table instead of a searchsorted
        # over the halo per window slot (27 lookups per cell per level: the sharded forward was slower
        # than the single-device one at 204 M points, 2026-10-30)
        pos_of_global = np.full((S, sizes[lvl]), junk, dtype=np.int32)
        for sidx in range(S):
            pos_of_global[sidx, sidx * block_coarse : (sidx + 1) * block_coarse] = np.arange(block_coarse, dtype=np.int32)
            pos_of_global[sidx, halos[sidx]] = block_coarse + np.arange(len(halos[sidx]), dtype=np.int32)
        put = partial(jax.device_put, device=NamedSharding(mesh, P(ax)))
        levels.append(
            _ShardLevel(
                True, False, block_cells, block_coarse, block_fine, H, bis, bfs,
                put(jnp.asarray(halo_ids, dtype=jnp.int32)), put(jnp.asarray(exp, dtype=jnp.int32)),
                put(jnp.asarray(imp_src, dtype=jnp.int32)), put(jnp.asarray(imp_pos, dtype=jnp.int32)), put(jnp.asarray(imp_slot, dtype=jnp.int32)),
                put(jnp.asarray(pos_of_global)),
            )
        )
    return tuple(levels)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["base_dist", "base_coords", "levels", "shard_levels"],
    meta_fields=["level_shapes", "window_size", "compress", "chunk_size", "n_shards", "max_distance", "min_distance"],
)
@dataclass(frozen=True)
class ICRStructure:
    """The grid-dependent part of an :class:`ICRGP`, independent of the kernel:
    the compressed refinement windows of every level (unique window
    coordinates and each cell's representative), the base-level distances and
    coordinates and, when built for a mesh, the per-shard halo tables.
    Build it once with :func:`build_icr_structure` (or
    :meth:`~nifty.re.ICRSpace.build_structure`) and pass it to every
    ``ICRGP`` / ``ICRSpace`` on the same grid: one compression, one copy of
    the tables on the devices, shared by all fields.
    """

    level_shapes: Tuple[Tuple[int, ...], ...]
    window_size: Any
    compress: Optional[tuple]
    chunk_size: int
    n_shards: int
    max_distance: float
    min_distance: float
    base_dist: jnp.ndarray
    base_coords: jnp.ndarray
    levels: Tuple[_Level, ...]
    shard_levels: Optional[Tuple[_ShardLevel, ...]]

    @property
    def n_unique(self) -> Tuple[int, ...]:
        return tuple(l.n_unique for l in self.levels)


def build_icr_structure(
    grid: Grid,
    *,
    window_size=None,
    compress: Optional[dict] = dict(rtol=1e-5, atol=1e-10, buffer_size=10_000),
    chunk_size=2**20,
    kernel_chunk=4096,
    mesh: Optional[Mesh] = None,
) -> ICRStructure:
    """Compress the refinement windows of `grid` (see :class:`ICRGP` for the
    parameters) and, with a one-dimensional `mesh`, build the per-shard halo
    tables; the result is shared by every ICR model on this grid."""
    if mesh is not None and len(mesh.axis_names) != 1:
        raise ValueError(f"mesh must be one-dimensional (one sharded axis per field), got axes {mesh.axis_names}")
    icr_kernel = ICRKernel(grid, None, window_size=window_size)
    compress_t = None if not compress else tuple(sorted(dict(compress).items()))
    compress_d = None if compress_t is None else dict(compress_t)
    # the index arithmetic of the grids is slow op by op; always run it under jit
    base_dist = jax.jit(lambda i: icr_kernel.get_kernel_distmat(i, None))(jnp.array([-1]))
    _, ((base_ids, _),) = icr_kernel.get_output_input_indices(jnp.array([-1]), None)
    base_coords = jax.jit(grid.at(0).index2coord)(jnp.asarray(base_ids))  # (dim, n0)
    levels = []
    for lvl in range(grid.depth):
        index = np.asarray(grid.at(lvl).refined_indices())
        index_shape = tuple(int(s) for s in index.shape[1:])
        cells = index.reshape(index.shape[0], -1)
        if compress_d is not None:
            uids, invids = compress_indices(icr_kernel, cells, lvl, **compress_d)
        else:
            uids, invids = cells, np.arange(cells.shape[1])
        ucoords = jax.jit(lambda i, lvl=lvl: _window_coordinates(icr_kernel, i, lvl))(jnp.asarray(uids))
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
    levels = tuple(levels)
    level_shapes = tuple(tuple(int(s) for s in grid.at(l).shape) for l in range(grid.depth + 1))
    bd = np.asarray(base_dist)
    d_max, d_min = float(np.max(bd)), float(np.min(bd[bd > 0])) if np.any(bd > 0) else np.inf
    for l in levels:
        for start in range(0, l.n_unique, int(kernel_chunk)):
            dd = np.asarray(jax.jit(_distances)(l.ucoords[start : start + int(kernel_chunk)]))
            d_max = max(d_max, float(np.max(dd)))
            if np.any(dd > 0):
                d_min = min(d_min, float(np.min(dd[dd > 0])))
    shard_levels = None if mesh is None else _icr_shard_tables(icr_kernel, levels, level_shapes, mesh, int(chunk_size))
    if mesh is not None:  # the replicated tables live on the mesh too (not on device 0 alone)
        rep = partial(jax.device_put, device=NamedSharding(mesh, P()))
        base_dist, base_coords = rep(base_dist), rep(base_coords)
        levels = tuple(dataclasses.replace(l, invids=rep(l.invids), ucoords=rep(l.ucoords)) for l in levels)
    return ICRStructure(
        level_shapes=level_shapes, window_size=icr_kernel.window_size, compress=compress_t, chunk_size=int(chunk_size),
        n_shards=1 if mesh is None else int(mesh.shape[mesh.axis_names[0]]), max_distance=d_max, min_distance=d_min,
        base_dist=base_dist, base_coords=base_coords, levels=levels, shard_levels=shard_levels,
    )


class ICRGP(KernelGP):
    """Gaussian process on a multigrid, generated by iterative charted
    refinement; the field lives on the finest level, the excitations are a
    list of arrays, one per level in the level's grid shape (coarsest first).

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
        Number of cells refined at once. The reverse pass accumulates every
        chunk's cotangent into the full coarse level, so few large chunks are
        much cheaper than many small ones (204 M points on one A100: vjp 2.2 s
        with 2^16, 0.22 s with 2^20, same memory within 2 %).
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
    _shard_levels: Optional[Tuple[_ShardLevel, ...]] = field(metadata=dict(static=False))
    mesh: Optional[Mesh] = field(metadata=dict(static=True))

    def __init__(
        self,
        grid: Grid,
        kernel,
        *,
        window_size=None,
        compress: Optional[dict] = dict(rtol=1e-5, atol=1e-10, buffer_size=10_000),
        chunk_size=2**20,
        kernel_chunk=4096,
        cholesky_jitter=1e-6,
        prefix="",
        radial_bins=None,
        dtype=jnp.float64,
        mesh: Optional[Mesh] = None,
        structure: Optional[ICRStructure] = None,
    ):
        self.multigrid = grid
        if mesh is not None and len(mesh.axis_names) != 1:
            raise ValueError(f"mesh must be one-dimensional (one sharded axis per field), got axes {mesh.axis_names}")
        self.mesh = mesh
        self.chunk_size = int(chunk_size)
        self.kernel_chunk = int(kernel_chunk)
        self.cholesky_jitter = float(cholesky_jitter)
        level_shapes = tuple(tuple(int(s) for s in grid.at(l).shape) for l in range(grid.depth + 1))
        if structure is None:
            structure = build_icr_structure(grid, window_size=window_size, compress=compress, chunk_size=chunk_size, kernel_chunk=kernel_chunk, mesh=mesh)
        else:
            # a shared structure: the grid must be the same; its window size and compression take precedence
            if tuple(structure.level_shapes) != level_shapes:
                raise ValueError(f"the ICRStructure was built for levels {structure.level_shapes}, this grid has {level_shapes}")
            S = 1 if mesh is None else int(mesh.shape[mesh.axis_names[0]])
            if mesh is not None and (structure.shard_levels is None or structure.n_shards != S):
                logger.info(f"ICRGP: the shared ICRStructure has no tables for {S} shards; building them")
                structure = dataclasses.replace(
                    structure,
                    shard_levels=_icr_shard_tables(ICRKernel(grid, None, window_size=structure.window_size), structure.levels, level_shapes, mesh, self.chunk_size),
                    n_shards=S,
                )
        self.icr_kernel = ICRKernel(grid, None, window_size=structure.window_size)
        self._compress = structure.compress
        self._base_dist, self._base_coords = structure.base_dist, structure.base_coords
        self._levels = structure.levels
        self._shard_levels = structure.shard_levels if mesh is not None else None
        self._level_shapes = level_shapes
        sizes = [int(np.prod(sh)) for sh in self._level_shapes]
        self._level_offsets = tuple(int(o) for o in np.cumsum([0] + sizes))
        self.max_distance, self.min_distance = structure.max_distance, structure.min_distance
        d_max, d_min = self.max_distance, self.min_distance
        super().__init__(
            kernel, radial_bins=radial_bins, shape=self._level_shapes[-1], excitation_shapes=self._level_shapes, prefix=prefix,
            dtype=dtype, allow_callable=True,
        )

        if self.radial_bins is not None and d_max > float(self.radial_bins[-1]):
            logger.warning(
                f"the covariance table ends at {float(self.radial_bins[-1]):.4g} but the refinement windows reach"
                f" distance {d_max:.4g}; the covariance is zero beyond the table"
            )
        if self.radial_bins is not None and self.radial_bins.shape[0] > 1 and d_min < float(self.radial_bins[1]):
            logger.warning(
                f"the covariance table's first bin is at {float(self.radial_bins[1]):.4g} but distinct points are"
                f" {d_min:.4g} apart: separations below the first bin interpolate towards the nugget-inflated C(0)"
                f" and can make the window covariances indefinite (NaN); choose r_min <= {0.5 * d_min:.4g}"
            )

    @property
    def compress(self) -> Optional[dict]:
        """Settings of the window compression (``None``: no compression)."""
        return None if self._compress is None else dict(self._compress)

    @property
    def structure(self) -> ICRStructure:
        """The grid-dependent tables of this model (compressed windows, base level,
        shard tables), to be passed to other ICR models on the same grid."""
        return ICRStructure(
            level_shapes=self._level_shapes, window_size=self.icr_kernel.window_size, compress=self._compress, chunk_size=self.chunk_size,
            n_shards=self.n_shards, max_distance=self.max_distance, min_distance=self.min_distance,
            base_dist=self._base_dist, base_coords=self._base_coords, levels=self._levels, shard_levels=self._shard_levels,
        )

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
        window_cov = self._window_covariance(covariance)
        base = jnp.linalg.cholesky(self._jittered(window_cov(self._base_coords)))
        mats = tuple(self._window_matrices(window_cov, l.ucoords, l.n_f) for l in self._levels)
        return base, mats

    def _window_covariance(self, covariance):
        """``window_cov(coords)``: the covariance matrices of windows given by their
        coordinates, from the table (interpolated at the distances) or the covariance function."""
        if self.coordinate_kernel:
            k = covariance
            k = vmap(k, in_axes=(None, -1), out_axes=-1)
            k = vmap(k, in_axes=(-1, None), out_axes=-1)
            return lambda c: jnp.vectorize(lambda cc: k(cc, cc), signature="(d,n)->(n,n)")(c)
        rs = self.radial_bins
        return lambda c: jnp.interp(_distances(c), rs, covariance, right=0.0)

    def _window_matrices(self, window_cov, ucoords, n_f):
        """Refinement matrices ``(olf, L_ff)`` of the windows `ucoords` (K, dim, n),
        built in chunks of ``kernel_chunk`` with recomputation in the backward pass."""
        K = ucoords.shape[0]
        chunk = min(self.kernel_chunk, K)
        n_chunks = -(-K // chunk)
        pad = n_chunks * chunk - K
        coords = jnp.concatenate((ucoords, jnp.repeat(ucoords[-1:], pad, axis=0)))
        coords = coords.reshape((n_chunks, chunk) + ucoords.shape[1:])

        @jax.checkpoint
        def build(c):
            # keep XLA from folding the distance tables of all windows into constants
            c = jax.lax.optimization_barrier(c)
            return vmap(partial(refinement_matrices_cholesky, n_fsz=n_f))(self._jittered(window_cov(c)))

        olf, lff = jax.lax.map(build, coords)
        return olf.reshape((-1,) + olf.shape[2:])[:K], lff.reshape((-1,) + lff.shape[2:])[:K]

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
        """`xi`: the per-level excitation arrays (list, shapes ``batch + level shape``)."""
        if self.mesh is not None:
            return self._generate_sharded(xi, cov_vals)
        base, mats = self.matrices(cov_vals)
        batch = self.batch_shape(xi)
        xs = [x.reshape((-1,) + sh) for x, sh in zip(xi, self._level_shapes)]

        def single(*xs):
            f = (base @ xs[0].ravel()).reshape(self._level_shapes[0])
            for lvl, (olf, lff) in enumerate(mats):
                f = self._refine(lvl, f, xs[lvl + 1], olf, lff)
            return f

        return vmap(single)(*xs).reshape(batch + self.shape)

    # ---- sharded generation: one shard per contiguous block along the first grid axis
    @property
    def n_shards(self) -> int:
        return 1 if self.mesh is None else int(self.mesh.shape[self.mesh.axis_names[0]])

    @property
    def axis(self) -> Optional[str]:
        return None if self.mesh is None else self.mesh.axis_names[0]

    def _build_shard_tables(self):
        self._shard_levels = _icr_shard_tables(self.icr_kernel, self._levels, self._level_shapes, self.mesh, self.chunk_size)

    @property
    def level_sharded(self):
        """Per level: whether it is sharded along its first axis (else replicated)."""
        return (False,) + tuple(t.sharded for t in self._shard_levels)

    @property
    def shardings(self):
        """Per level: the sharding of its excitation array (and of the field at that level)."""
        return tuple(
            NamedSharding(self.mesh, P(self.axis, *([None] * (len(sh) - 1))) if sharded else P())
            for sharded, sh in zip(self.level_sharded, self._level_shapes)
        )

    @property
    def sharding(self):
        """Sharding of the output (the finest level)."""
        return self.shardings[-1]

    def shard(self, x):
        """Places the per-level excitations of a parameter dict on the mesh."""
        x = dict(x)
        x[self.xi_name] = [jax.device_put(np.asarray(a), s) for a, s in zip(x[self.xi_name], self.shardings)]
        return x

    def _generate_sharded(self, xi, cov_vals):
        S, ax = self.n_shards, self.axis
        # the covariance table enters the shard_map body as an argument (replicated), not through a
        # closure: a closed-over value placed on the mesh (replicated hyperparameters of the model)
        # carries the mesh's sharding into the body, where the context mesh is the manual one and
        # XLA rejects the mix; a coordinate kernel (a callable) stays a closure
        cov_arg = None if self.coordinate_kernel else cov_vals
        # the unique windows are distributed over the shards: every shard builds its share of the
        # refinement matrices and the shares are all-gathered (the kernel build dominates the ICR cost)
        padded = []
        for l in self._levels:
            K = l.n_unique
            Ks = -(-K // S)
            uc = jnp.concatenate((l.ucoords, jnp.repeat(l.ucoords[-1:], Ks * S - K, axis=0)))
            padded.append(uc.reshape((S, Ks) + l.ucoords.shape[1:]))
        n_fs = tuple(l.n_f for l in self._levels)
        n_uniques = tuple(l.n_unique for l in self._levels)
        shard_levels = self._shard_levels
        spec = jax.tree_util.tree_map(lambda _: P(ax), shard_levels)

        def local(xs, cov, coords, tabs):
            # xs: the per-level excitations, the block of the level for sharded levels, full otherwise
            sidx = jax.lax.axis_index(ax)
            window_cov = self._window_covariance(cov_vals if cov is None else cov)
            base = jnp.linalg.cholesky(self._jittered(window_cov(self._base_coords)))  # replicated (small)
            mats = []
            for uc, n_f, K in zip(coords, n_fs, n_uniques):
                olf, lff = self._window_matrices(window_cov, uc[0], n_f)
                olf = jax.lax.all_gather(olf, ax, axis=0, tiled=True)[:K]
                lff = jax.lax.all_gather(lff, ax, axis=0, tiled=True)[:K]
                mats.append((olf, lff))
            f = (base @ xs[0].ravel()).reshape(self._level_shapes[0])  # level 0: replicated dense generation
            for lvl, (olf, lff) in enumerate(mats):
                T = tabs[lvl]
                if not T.sharded:  # replicated fine level
                    f = self._refine(lvl, f, xs[lvl + 1], olf, lff)
                    continue
                fb = f.reshape(-1)
                if T.coarse_full:
                    ext = fb
                else:
                    gathered = jax.lax.all_gather(fb[T.exp[0]], ax, axis=0, tiled=False)  # (S, E)
                    ext = jnp.concatenate([fb, jnp.zeros((T.halo_size + 1,), fb.dtype)])
                    ext = ext.at[T.imp_slot[0]].set(gathered[T.imp_src[0], T.imp_pos[0]])
                f = self._refine_block(lvl, T, sidx, ext, xs[lvl + 1].reshape(-1), olf, lff)
            return f

        xi_specs = tuple(P(ax, *([None] * (len(sh) - 1))) if sharded else P() for sharded, sh in zip(self.level_sharded, self._level_shapes))
        in_specs = (xi_specs, None if cov_arg is None else P(), tuple(P(ax) for _ in padded), spec)
        f = jax.shard_map(local, mesh=self.mesh, in_specs=in_specs, out_specs=P(ax), check_vma=False)
        batch = self.batch_shape(xi)
        if len(batch) == 0:
            return f(tuple(xi), cov_arg, tuple(padded), shard_levels)
        # shard_map maps the first axis of every input onto the mesh: run leading batch axes outside
        xs = [x.reshape((-1,) + sh) for x, sh in zip(xi, self._level_shapes)]
        return vmap(lambda *x: f(x, cov_arg, tuple(padded), shard_levels))(*xs).reshape(batch + self.shape)

    def _refine_block(self, lvl, T, sidx, ext, xi_fine, olf, lff):
        """Refinement of one shard's block of cells at level `lvl` from the extended
        coarse array `ext` (block + halo, or the full coarse level) and the block's
        fine excitations; same chunked map as :meth:`_refine`."""
        L = self._levels[lvl]
        n = T.block_cells
        chunk = min(self.chunk_size, n)
        n_chunks = -(-n // chunk)
        pad = n_chunks * chunk - n
        cell0 = sidx * n
        inv = jax.lax.dynamic_slice_in_dim(L.invids, cell0, n)
        kid = jnp.concatenate((inv, jnp.repeat(inv[-1:], pad))).reshape(n_chunks, chunk)
        starts = jnp.arange(n_chunks) * chunk
        index_lo = jnp.asarray(L.index_lo)
        cshape = self._level_shapes[lvl]
        fshape = self._level_shapes[lvl + 1]
        coarse0 = 0 if T.coarse_full else sidx * T.block_coarse
        fine0 = sidx * T.block_fine
        halo_ids = None if T.coarse_full else T.halo_ids[0]
        pos_of_global = None if T.coarse_full or T.pos_of_global is None else T.pos_of_global[0]

        @jax.checkpoint
        def body(start, kid):
            flat = cell0 + jnp.minimum(start + jnp.arange(chunk), n - 1)
            cells = jnp.stack(jnp.unravel_index(flat, L.index_shape)) + index_lo[:, jnp.newaxis]
            _, ((gc, _), (gf, _)) = self.icr_kernel.get_output_input_indices(cells, lvl)
            gidx = jnp.ravel_multi_index(tuple(gc), cshape, mode="clip")
            if halo_ids is None:
                pos = gidx
            elif pos_of_global is not None:
                pos = pos_of_global[gidx]  # one gather per window slot
            else:  # tables of an older structure without the position table
                own = (gidx >= coarse0) & (gidx < coarse0 + T.block_coarse)
                pos = jnp.where(own, gidx - coarse0, T.block_coarse + jnp.searchsorted(halo_ids, gidx))
            xc = ext[pos]
            fidx = jnp.ravel_multi_index(tuple(gf), fshape, mode="clip") - fine0
            xf = xi_fine[fidx]
            return jnp.einsum("cfw,cw->cf", olf[kid], xc) + jnp.einsum("cfg,cg->cf", lff[kid], xf)

        res = jax.lax.map(lambda xs: body(*xs), (starts, kid))
        res = res.reshape(-1, L.n_f)[:n]
        nd = len(L.index_shape)
        res = res.reshape(T.block_index_shape + L.splits)
        order = sum(((i, nd + i) for i in range(nd)), ())
        return res.transpose(order).reshape(T.block_fine_shape)
