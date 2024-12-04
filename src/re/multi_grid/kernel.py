#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Philipp Frank, Gordian Edenhofer

import operator
from collections import namedtuple
from functools import partial, reduce
from typing import Callable, Iterable, Tuple

import jax.numpy as jnp
import numpy as np
from jax import eval_shape, jit, vmap
from jax.lax import scan
from numpy import typing as npt

from ..num import amend_unique_
from ..refine.util import refinement_matrices
from .grid import FlatGrid, Grid, OpenGridAtLevel
from .grid_impl import HEALPixGridAtLevel


def _dist(x, y, periodicity=None):
    d = y - x
    if periodicity is not None:
        p = np.atleast_1d(periodicity)[(slice(None),) + (np.newaxis,) * (d.ndim - 1)]
        return jnp.where(jnp.abs(d) > p / 2.0, d - jnp.sign(d) * p, d)
    return d


_IdxMap = namedtuple("_IdxMap", ("shift", "index2flatindex"))


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


class Kernel:
    """
    - A structure on a multigrid that can apply linear operators with inputs accross
      the grid
    - A Kernel could be as simple as coarse graining children to a parent; or could
      be a full ICRefine or MSCrefine
    - Should be fully jax transformable to allow seamlessly being used in larger
      models (e.g. models for variable kernels should be able to output an instance
      of 'kernel')
    """

    def __init__(self, grid):
        self._grid = grid

    @property
    def grid(self) -> Grid:
        return self._grid

    def get_output_input_indices(
        self, index, level: int
    ) -> Tuple[
        Tuple[npt.NDArray[np.int_], int], Tuple[Tuple[npt.NDArray[np.int_], int], ...]
    ]:
        raise NotImplementedError()

    def get_matrices(self, index, level):
        raise NotImplementedError()

    def compress(
        self,
        *,
        rtol=1e-5,
        atol=1e-10,
        buffer_size=10000,
        use_distances=True,
        distance_norm=partial(jnp.linalg.norm, axis=0),
    ):
        """Compress the kernel matrices for the given grid.

        This links kernel matrices by similarity and subsequently looks them up via
        tables. Compressing the kernel in this way has the potential to drastically
        reduce the memory requirements of applying it and make the inference of
        kernels faster.
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
                f = get_distance_matrices if use_distances else self.get_matrices
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
        return CompressedKernel(self, uindices, invindices, indexmaps)


class CompressedKernel(Kernel):
    def __init__(self, kernel: Kernel, uindices, invindices, indexmaps):
        self._get_output_input_indices = kernel.get_output_input_indices
        self.uindices = uindices
        self.invindices = invindices
        self.indexmaps = indexmaps
        self._kernels = tuple(
            kernel.get_matrices(ii, ll) for ll, ii in enumerate(uindices)
        )
        self._base_kernel = kernel.get_matrices(jnp.array([-1]), None)
        super().__init__(grid=kernel.grid)

    def replace_kernel(self, kernel):
        """Replace the kernel and recomput the kernel matrices while keeping the
        tables for speedy lookups fixed.

        This is useful for updating the covariance function of a kernel without
        recreating the tables in an expensive optimization.
        """
        return self.__class__(kernel, self.uindices, self.invindices, self.indexmaps)

    def get_output_input_indices(self, index, level):
        return self._get_output_input_indices(index, level)

    def get_matrices(self, index, level):
        if level is None:
            return self._base_kernel
        index = self.indexmaps[level].index2flatindex(index)[0]
        index = self.invindices[level][index - self.indexmaps[level].shift]
        return tuple(kk[index] for kk in self._kernels[level])


def _suitable_window_size(grid_at_level, default=3) -> Tuple[int]:
    wsz = []
    for g in grid_at_level.raw_grids:
        if isinstance(g, HEALPixGridAtLevel):  # special-case HEALPix
            wsz += [9]
        elif isinstance(g, OpenGridAtLevel):
            wsz += list(g.padding * 2 + 1)
        else:
            wsz += [default] * g.ndim
    return tuple(wsz)


class ICRKernel(Kernel):
    def __init__(self, grid, covariance, *, window_size=None):
        self._covariance_elem = covariance
        k = self._covariance_elem
        k = vmap(k, in_axes=(None, -1), out_axes=-1)
        k = vmap(k, in_axes=(-1, None), out_axes=-1)
        self._covariance_outer = k
        if window_size is None:
            window_size = tuple(
                _suitable_window_size(grid.at(lvl)) for lvl in range(grid.depth)
            )
        self._window_size = tuple(
            np.broadcast_to(window_size, (grid.depth,) + grid.at(0).shape.shape)
        )
        super().__init__(grid=grid)

    @property
    def covariance_outer(self):
        return self._covariance_outer

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

    def get_matrices(self, index, level):
        from ..refine.util import sqrtm

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
