#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Philipp Frank, Gordian Edenhofer

import operator
from collections import namedtuple
from functools import partial, reduce
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax import eval_shape, jit, vmap
from jax.lax import scan
from numpy import typing as npt

from ..num import amend_unique_
from .grid import FlatGrid, Grid, OpenGridAtLevel
from .grid_impl import HEALPixGridAtLevel
from ..tree_math import solve, sqrtm


def apply_kernel(x, *, kernel):
    """Applies the kernel to values on an entire multigrid"""
    if len(x) != (kernel.grid.depth + 1):
        msg = f"input depth {len(x)} does not match grid depth {kernel.grid.depth}"
        raise ValueError(msg)
    for lvl, xx in enumerate(x):
        g = kernel.grid.at(lvl)
        if xx.size != g.size:
            msg = f"input at level {lvl} of size {xx.size} does not match grid of size {g.size}"
            raise ValueError(msg)

    def apply_at(index, level, x):
        assert index.ndim == 1
        iout, iin = kernel.get_output_input_indices(index, level)
        kernels = kernel.get_matrices(index, level)
        assert len(iin) == len(kernels)
        res = reduce(
            operator.add,
            (kk @ x[x_lvl][tuple(idx)] for kk, (idx, x_lvl) in zip(kernels, iin)),
        )
        return iout, res.reshape(iout[0].shape[1:])

    x = list(x)
    _, x[0] = apply_at(jnp.array([-1]), None, x)  # Use dummy index for base
    for lvl in range(kernel.grid.depth):
        g = kernel.grid.at(lvl)
        index = g.refined_indices()
        # TODO this selects window for each index individually
        f = apply_at
        for i in range(g.ndim):
            f = vmap(f, (1, None, None), ((g.ndim - i, None), g.ndim - i - 1))
        (_, lvl_nxt), res = f(index, lvl, x)
        x[lvl_nxt] = kernel.grid.at(lvl_nxt).resort(res)
    return x


_IdxMap = namedtuple("_IdxMap", ("shift", "index2flatindex"))
_CompressedIndexMap = namedtuple(
    "_CompressedIndexMap",
    ("base_kernel", "kernels", "uindices", "indexmaps", "invindices"),
)


class Kernel:
    """
    - Apply linear operators with inputs accross an arbitrary grid
    - A Kernel could be as simple as coarse graining children to a parent; or a full
      ICR or MSC implementation
    - Fully jax transformable to allow seamlessly being used in larger models
    """

    def __init__(self, grid, *, _cim: Optional[_CompressedIndexMap] = None):
        self._grid = grid
        self._cim = _cim

    def replace(self, *, _cim=None, **kwargs):
        _cim = self._cim if _cim is None else _cim
        return self.__class__(self.grid, **kwargs, _cim=_cim)

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def compressed(self) -> bool:
        return self._cim is not None

    def get_output_input_indices(
        self, index, level: int
    ) -> Tuple[
        Tuple[npt.NDArray[np.int_], int], Tuple[Tuple[npt.NDArray[np.int_], int], ...]
    ]:
        raise NotImplementedError()

    def get_matrices(self, index, level):
        """Compute or lookup kernel matrices."""
        if self.compressed:
            return self.lookup_matrices(index, level)
        return self.compute_matrices(index, level)

    def compute_matrices(self, index, level):
        """Compute kernel matrices from scratch."""
        raise NotImplementedError()

    def lookup_matrices(self, index, level):
        """Efficient retrieval of kernel matrices for a compressed kernel."""
        if self._cim is None:
            msg = "kernel needs to be compressed first for fast lookups"
            raise NotImplementedError(msg)

        if level is None:
            return self._cim.base_kernel
        index = self._cim.indexmaps[level].index2flatindex(index)[0]
        index = self._cim.invindices[level][index - self._cim.indexmaps[level].shift]
        return tuple(kk[index] for kk in self._cim.kernels[level])

    def compress_indices(
        self,
        *,
        rtol=1e-5,
        atol=1e-10,
        buffer_size=10000,
        use_distances=True,
        distance_norm=partial(jnp.linalg.norm, axis=0),
    ):
        """Link kernel matrices to preserve memory and increase speed.

        You probably don't want to access this function directly. Instead use
        `compress` or `replace` and `compress_matrices`.
        """

        def get_distance_matrices(index, level):
            if level is None:
                raise NotImplementedError
            (out, olvl), ids = self.get_output_input_indices(index, level)
            out = out.reshape(index.shape + (-1,))
            out = self.grid.at(olvl).index2coord(out)
            assert index.ndim == out.ndim - 1
            ids = tuple(self.grid.at(ii[1]).index2coord(ii[0]) for ii in ids)
            ids = jnp.concatenate(ids, axis=-1)
            assert index.ndim == ids.ndim - 1
            return (distance_norm(out[..., jnp.newaxis] - ids[..., jnp.newaxis, :]),)

        gridf = self.grid if isinstance(self.grid, FlatGrid) else FlatGrid(self.grid)
        uindices = []
        invindices = []
        indexmaps = []
        for lvl in range(self.grid.depth):
            grid_at = self.grid.at(lvl)
            gridf_at = gridf.at(lvl)

            def get_matrices(idx):
                f = get_distance_matrices if use_distances else self.compute_matrices
                ker = f(gridf_at.flatindex2index(jnp.atleast_1d(idx)), lvl)
                ker = jnp.concatenate(tuple(kk.ravel() for kk in ker))
                return ker

            @jit
            def scanned_amend_unique(carry, idx, shift):
                (u, inv) = carry
                k = get_matrices(idx)
                u, invid = amend_unique_(u, k, axis=0, atol=atol, rtol=rtol)
                inv = inv.at[idx - shift].set(invid)
                return (u, inv), invid

            indices = grid_at.refined_indices()
            indices = gridf_at.index2flatindex(indices)[0].ravel()
            shift = np.min(indices)
            size = np.max(indices) - shift
            inv = np.full(size, buffer_size + 1)

            shp = eval_shape(get_matrices, indices[0]).shape
            unique = jnp.full((buffer_size,) + shp, jnp.nan)

            (unique, inv), invid = scan(
                partial(scanned_amend_unique, shift=shift), (unique, inv), indices
            )
            _, idx = np.unique(invid, return_index=True)
            n = idx.size
            if n >= unique.shape[0] or not np.all(np.isnan(unique[n:])):
                raise ValueError("`mat_buffer_size` too small")
            uids = indices[idx]
            uids = gridf_at.flatindex2index(uids[np.newaxis, :])
            uindices.append(uids)
            invindices.append(inv)
            indexmaps.append(_IdxMap(shift, gridf_at.index2flatindex))

        return self.replace(
            _cim=_CompressedIndexMap(
                base_kernel=None,
                kernels=None,
                uindices=uindices,
                indexmaps=indexmaps,
                invindices=invindices,
            )
        )

    def compress_matrices(self):
        """Recompute the kernel matrices while keeping the tables for speedy lookups
        fixed.

        This is useful for updating the covariance function of a kernel without
        recreating the tables in an expensive optimization.
        """
        assert self._cim is not None

        base_kernel = self.compute_matrices(jnp.array([-1]), None)
        kernels = tuple(
            self.compute_matrices(ii, ll) for ll, ii in enumerate(self._cim.uindices)
        )
        cim = self._cim._replace(base_kernel=base_kernel, kernels=kernels)
        return self.replace(_cim=cim)

    def compress(self, *args, **kwargs):
        """Optimize the kernel matrices for the given grid.

        This links kernel matrices by similarity and subsequently looks them up via
        tables. Compressing the kernel in this way has the potential to drastically
        reduce the memory requirements of applying it and make the inference of
        kernels faster.
        """
        return self.compress_indices(*args, **kwargs).compress_matrices()


def _default_window_size(grid_at_level, default=3) -> Tuple[int]:
    wsz = []
    for g in grid_at_level.raw_grids:
        if isinstance(g, HEALPixGridAtLevel):  # special-case HEALPix
            wsz += [9]
        elif isinstance(g, OpenGridAtLevel):
            wsz += list(g.padding * 2 + 1)
        else:
            wsz += [default] * g.ndim
    return tuple(wsz)


def refinement_matrices(cov, n_fsz: int):
    cov_ff = cov[-n_fsz:, -n_fsz:]
    cov_fc = cov[-n_fsz:, :-n_fsz]
    cov_cc = cov[:-n_fsz, :-n_fsz]

    olf = solve(cov_cc, cov_fc.T, matrix_eqn=True)
    return olf.T, sqrtm(cov_ff - cov_fc @ olf)


class ICRKernel(Kernel):
    """Full ICR implementation taking an arbitrary grid and a covariance function."""

    def __init__(self, grid, covariance, *, window_size=None, _cim=None):
        self._covariance_elem = covariance
        if window_size is None:
            window_size = tuple(
                _default_window_size(grid.at(lvl)) for lvl in range(grid.depth)
            )
        self._window_size = tuple(
            np.broadcast_to(window_size, (grid.depth, grid.at(0).ndim))
        )
        super().__init__(grid=grid, _cim=_cim)

    def replace(self, *, covariance=None, window_size=None, _cim=None):
        cim = self._cim if _cim is None else _cim
        if covariance is not None and cim is not None:
            cim = cim._replace(base_kernel=None, kernels=None)
        else:
            covariance = self._covariance_elem
        window_size = self.window_size if window_size is None else window_size
        return self.__class__(self.grid, covariance, window_size=window_size, _cim=cim)

    @property
    def covariance_outer(self):
        k = self._covariance_elem
        k = vmap(k, in_axes=(None, -1), out_axes=-1)
        k = vmap(k, in_axes=(-1, None), out_axes=-1)
        return k

    @property
    def window_size(self):
        return self._window_size

    def get_output_input_indices(self, index, level):
        if level is None:
            grid_at_lvl = self.grid.at(0)
            pixel_indices = np.mgrid[tuple(slice(0, sz) for sz in grid_at_lvl.shape)]
            assert pixel_indices.shape[0] == grid_at_lvl.ndim
            return (pixel_indices, 0), (
                (pixel_indices.reshape(grid_at_lvl.ndim, -1), 0),
            )
        g = self.grid.at(level)
        assert index.shape[0] == g.ndim
        gc = g.neighborhood(index, self.window_size[level]).reshape(index.shape + (-1,))
        gout = g.children(index)
        gf = gout.reshape(index.shape + (-1,))
        return (gout, level + 1), ((gc, level), (gf, level + 1))

    def compute_matrices(self, index, level):
        if level is None:
            _, ((ids, _),) = self.get_output_input_indices(index, None)
            gc = self.grid.at(0).index2coord(ids)
            cov = self.covariance_outer(gc, gc)
            assert cov.shape == (gc.shape[1],) * 2
            return (sqrtm(cov),)

        _, ((idc, _), (idf, _)) = self.get_output_input_indices(index, level)

        def get_mat(gc, gf):
            gc = self.grid.at(level).index2coord(gc)
            gf = self.grid.at(level + 1).index2coord(gf)
            assert gc.shape[0] == gf.shape[0]
            assert gc.ndim == gf.ndim == 2
            coord = jnp.concatenate((gc, gf), axis=-1)
            cov = self.covariance_outer(coord, coord)
            return refinement_matrices(cov, gf.shape[1])

        f = get_mat
        for _ in range(index.ndim - 1):
            f = vmap(f, in_axes=(1, 1))
        return f(idc, idf)
