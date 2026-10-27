#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Philipp Frank, Gordian Edenhofer

"""Iterative charted refinement (ICR) as a factor of a correlated field.

:class:`ICRSpace` holds a multigrid (:class:`~nifty.re.multi_grid.Grid`) and
the settings of the refinement: the window sizes, the compression of the
refinement matrices and the range of the covariance table. Given the
covariance on its radial bins, it generates the field level by level, each
level in chunks of cells (``chunk_size`` cells at a time, recomputed in the
backward pass), so that the memory of an application is bounded by the field
and one chunk of gathered windows regardless of the grid's axes.

The refinement matrices come from one Cholesky factorization per unique
window covariance (``olf = L_fc L_cc^-1``, ``L_ff``); the unique windows are
found by hashing the distance matrices quantized on the compression
tolerance, followed by a greedy merge of the representatives.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from ..correlated_field import KernelSpace
from ..logger import logger
from ..num import amend_unique_
from .grid import Grid
from .kernel import ICRKernel

__all__ = ["ICRSpace", "compress_indices", "refinement_matrices_cholesky"]


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


class ICRSpace(KernelSpace):
    """Multigrid factor generated by iterative charted refinement.

    Parameters
    ----------
    grid : nifty.re.multi_grid.Grid
        The multigrid; the field lives on its finest level.
    r_min, r_max, n_bins : float, float, int
        Covariance table (see :class:`~nifty.re.correlated_field.KernelSpace`).
        By default `r_max` is the largest and `r_min` half the smallest
        non-zero distance occurring in the refinement windows. The covariance
        is interpolated linearly between the (logarithmically spaced) radial
        bins; too few bins per decade make the interpolated window covariances
        indefinite and the Cholesky factorization fails with NaNs. For a steep
        spectrum (amplitude slope -3 in three dimensions) 35 bins per decade
        were indefinite and 135 per decade fine; keep at least 100 per decade
        of ``r_max / r_min``.
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
    jitter : float
        Relative inflation of the covariance at zero distance.
    radial_bins, k_bins : arrays, optional
        Explicit bins in place of `r_min`, `r_max` and `n_bins`.
    """

    _base_dist: jnp.ndarray = field(metadata=dict(static=False))
    _levels: Tuple[_Level, ...] = field(metadata=dict(static=False))

    def __init__(
        self,
        grid: Grid,
        *,
        r_min=None,
        r_max=None,
        n_bins=1024,
        window_size=None,
        compress: Optional[dict] = dict(rtol=1e-5, atol=1e-10, buffer_size=10_000),
        chunk_size=2**16,
        kernel_chunk=4096,
        cholesky_jitter=1e-6,
        jitter=1e-6,
        radial_bins=None,
        k_bins=None,
    ):
        self.multigrid = grid
        self.kernel = ICRKernel(grid, None, window_size=window_size)
        self.chunk_size = int(chunk_size)
        self.kernel_chunk = int(kernel_chunk)
        self.cholesky_jitter = float(cholesky_jitter)
        self._compress = None if not compress else tuple(sorted(dict(compress).items()))

        g0 = grid.at(0)
        dim = int(g0.index2coord(np.zeros((g0.ndim, 1), dtype=np.int64)).shape[0])
        # the index arithmetic of the grids is slow op by op; always run it under jit
        self._base_dist = jax.jit(lambda i: self.kernel.get_kernel_distmat(i, None))(jnp.array([-1]))
        levels = []
        for lvl in range(grid.depth):
            index = np.asarray(grid.at(lvl).refined_indices())
            index_shape = tuple(int(s) for s in index.shape[1:])
            cells = index.reshape(index.shape[0], -1)
            if self.compress is not None:
                uids, invids = compress_indices(self.kernel, cells, lvl, **self.compress)
            else:
                uids, invids = cells, np.arange(cells.shape[1])
            ucoords = jax.jit(lambda i, lvl=lvl: _window_coordinates(self.kernel, i, lvl))(jnp.asarray(uids))
            n_f = int(np.prod(grid.at(lvl).splits))
            logger.info(f"ICRSpace level {lvl}: {cells.shape[1]} cells, {uids.shape[1]} unique windows")
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

        if (r_max is None and radial_bins is None) or (r_min is None and (radial_bins is None or k_bins is None)):
            d_max, d_min = float(np.max(self._base_dist)), np.inf
            bd = np.asarray(self._base_dist)
            if np.any(bd > 0):
                d_min = float(bd[bd > 0].min())
            for l in levels:
                for start in range(0, l.n_unique, self.kernel_chunk):
                    d = np.asarray(jax.jit(_distances)(l.ucoords[start : start + self.kernel_chunk]))
                    d_max = max(d_max, float(d.max()))
                    if np.any(d > 0):
                        d_min = min(d_min, float(d[d > 0].min()))
            if r_max is None and radial_bins is None:
                r_max = d_max * (1.0 + 1e-3)
            if r_min is None and (radial_bins is None or k_bins is None):
                r_min = 0.5 * d_min
        super().__init__(
            self._level_shapes[-1],
            dim,
            n_points=sum(sizes),
            r_min=r_min,
            r_max=r_max,
            n_bins=n_bins,
            jitter=jitter,
            radial_bins=radial_bins,
            k_bins=k_bins,
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
        return self.kernel.window_size

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
        unique windows for the `covariance` on :attr:`radial_bins`."""
        rs = self.radial_bins

        def kf(d):
            return jnp.interp(d, rs, covariance, right=0.0)

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
                return vmap(partial(refinement_matrices_cholesky, n_fsz=n_f))(self._jittered(kf(_distances(c))))

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
            _, ((gc, _), (gf, _)) = self.kernel.get_output_input_indices(cells, lvl)
            xc = coarse[tuple(gc)]
            xf = xi_fine[tuple(gf)]
            return jnp.einsum("cfw,cw->cf", olf[kid], xc) + jnp.einsum("cfg,cg->cf", lff[kid], xf)

        res = jax.lax.map(lambda xs: body(*xs), (starts, kid))
        res = res.reshape(-1, L.n_f)[:n]
        nd = len(L.index_shape)
        res = res.reshape(L.index_shape + L.splits)
        order = sum(((i, nd + i) for i in range(nd)), ())
        return res.transpose(order).reshape(L.fine_shape)

    def apply_sqrt(self, xi, covariance):
        base, mats = self.matrices(covariance)

        def single(x):
            xs = self.split_excitations(x)
            f = (base @ xs[0].ravel()).reshape(self._level_shapes[0])
            for lvl, (olf, lff) in enumerate(mats):
                f = self._refine(lvl, f, xs[lvl + 1], olf, lff)
            return f

        return vmap(single)(xi)
