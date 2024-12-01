#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Philipp Frank, Gordian Edenhofer

import operator
from collections import namedtuple
from functools import partial, reduce
from typing import Callable, Iterable, Tuple

from numpy import typing as npt
import jax.numpy as jnp
import numpy as np
from jax import eval_shape, jit, vmap
from jax.lax import scan

from ..num import amend_unique_
from ..refine.util import refinement_matrices
from .indexing import FlatGrid, Grid


def _dist(x, y, periodicity=None):
    d = y - x
    if periodicity is not None:
        p = np.atleast_1d(periodicity)[(slice(None),) + (np.newaxis,) * (d.ndim - 1)]
        return jnp.where(jnp.abs(d) > p / 2.0, d - jnp.sign(d) * p, d)
    return d


_IdxMap = namedtuple("_IdxMap", ("shift", "index2flatindex"))


class KernelBase:
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

    def __call__(self, x, copy=True):
        """Applies the kernel to values on an entire multigrid"""
        if len(x) != (self.grid.depth + 1):
            msg = f"input depth {len(x)} does not match grid depth {self.grid.depth}"
            raise ValueError(msg)
        for lvl, xx in enumerate(x):
            g = self.grid.at(lvl)
            if xx.size != g.size:
                msg = f"input at level {lvl} of size {xx.size} does not match grid of size {g.size}"
                raise ValueError(msg)

        def apply_at(index, level, x):
            assert index.ndim == 1
            iout, iin = self.get_output_input_indices(index, level)
            kernels = self.get_matrices(index, level)
            assert len(iin) == len(kernels)
            res = reduce(
                operator.add,
                (kk @ x[x_lvl][tuple(idx)] for kk, (idx, x_lvl) in zip(kernels, iin)),
            )
            return iout, res.reshape(iout[0].shape[1:])

        x = list(jnp.copy(xx) for xx in x) if copy else x
        # Use dummy index for base
        _, x[0] = apply_at(jnp.array([-1]), None, x)
        for lvl in range(self.grid.depth):
            g = self.grid.at(lvl)
            index = g.refined_indices()
            # TODO this selects window for each index individually
            f = apply_at
            for i in range(g.ndim):
                f = vmap(f, (1, None, None), ((g.ndim - i, None), g.ndim - i - 1))
            (_, lvl_nxt), res = f(index, lvl, x)
            x[lvl_nxt] = self.grid.at(lvl_nxt).resort(res)
        return x

    def freeze(
        self,
        *,
        rtol=1e-5,
        atol=1e-10,
        buffer_size=10000,
        use_distances=True,
        uindices=None,
        invindices=None,
        indexmaps=None,
        distance_norm=partial(jnp.linalg.norm, axis=0),
    ):
        """Evaluate the kernel and store it in a `FrozenKernel`. Kernels may be
        grouped by similarity and accessed via lookup"""
        if uindices is not None and invindices is not None and indexmaps is not None:
            _FrozenKernel(self, uindices, invindices, indexmaps)

        def get_distance_matrix(index, level):
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
            grid_at_f = gridf.at(lvl)

            def get_matrices(idx):
                f = get_distance_matrix if use_distances else self.get_matrices
                ker = f(grid_at_f.flatindex2index(jnp.atleast_1d(idx)), lvl)
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
            indices = grid_at_f.index2flatindex(indices)[0].ravel()
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
            uids = grid_at_f.flatindex2index(uids[np.newaxis, :])
            uindices.append(uids)
            invindices.append(inv)
            indexmaps.append(_IdxMap(shift, grid_at_f.index2flatindex))
        return _FrozenKernel(self, uindices, invindices, indexmaps)


class _FrozenKernel(KernelBase):
    def __init__(self, kernel: KernelBase, uindices, invindices, indexmaps):
        self._get_refinement_indices = kernel.get_output_input_indices
        self.uindices = uindices
        self.invindices = invindices
        self.indexmaps = indexmaps
        self._kernels = tuple(
            kernel.get_matrices(ii, ll) for ll, ii in enumerate(uindices)
        )
        self._base_kernel = kernel.get_matrices(jnp.array([-1]), None)
        super().__init__(grid=kernel.grid)

    def get_output_input_indices(self, index, level):
        return self._get_refinement_indices(index, level)

    def get_matrices(self, index, level):
        if level is None:
            return self._base_kernel

        map_atlevel = self.indexmaps[level]
        inv_atlevel = self.invindices[level]
        index = map_atlevel.index2flatindex(index)[0]
        index = inv_atlevel[index - map_atlevel.shift]
        return tuple(kk[index] for kk in self._kernels[level])


class ICRefine(KernelBase):
    def __init__(self, grid, covariance, *, window_size):
        self._covariance_elem = covariance
        k = self._covariance_elem
        k = vmap(k, in_axes=(None, -1), out_axes=-1)
        k = vmap(k, in_axes=(-1, None), out_axes=-1)
        self._covariance_outer = k
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
        if (level >= self.grid.depth) or (level < -1):
            mg = f"Level {level} out of bounds for grid deph {self.grid.depth}"
            raise ValueError(mg)
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
            assert gc.ndim == 2
            cov = self.covariance_outer(gc, gc)
            assert cov.shape == (gc.shape[1],) * 2
            return (sqrtm(cov),)

        _, ((idc, _), (idf, _)) = self.get_output_input_indices(index, level)

        def _get_kernel(gc, gf):
            gc = self.grid.at(level).index2coord(gc)
            gf = self.grid.at(level + 1).index2coord(gf)
            assert gc.shape[0] == gf.shape[0]
            assert gc.ndim == gf.ndim == 2
            coord = jnp.concatenate((gc, gf), axis=-1)
            cov = self.covariance_outer(coord, coord)
            return refinement_matrices(cov, gf.shape[1])

        f = _get_kernel
        for _ in range(index.ndim - 1):
            f = vmap(f, in_axes=(1, 1))
        return f(idc, idf)


class FixedKernelFunctionBatch(KernelBase):
    def __init__(
        self, grid: Grid, window_size: Iterable[int], kernel: Callable, periodicity=None
    ):
        self._kernel = kernel
        self._window_size = window_size  # TODO different at different levels?
        self._periodicity = periodicity
        super().__init__(grid=grid)

    def get_output_input_indices(self, index, level):
        if level == 0:
            grid = np.mgrid[tuple(slice(0, s) for s in self.grid.shape)]
            grid = grid.reshape((grid.shape[0], -1))
            assert grid.shape[0] == index.shape[0]

            return (
                index[(slice(None),) * index.ndim + (1,)]
                * grid[(slice(None),) + (1,) * index.ndim + (slice(None),)]
            )
        targets = self.grid.at(level - 1).children(index).reshape(index.shape + (-1,))
        nbrs = (
            self.grid.at(level)
            .neighborhood(targets[..., -1], self._window_size)
            .reshape(index.shape + (-1,))
        )
        return (targets, level), ((nbrs, level),)

    def evaluate_kernel(self, out_index, in_index, level):
        grid_at_level = self.grid.at(level)
        targets_pos = grid_at_level.index2coord(out_index)
        nbrs_delta = grid_at_level.index2coord(in_index)
        targets_pos = targets_pos[..., jnp.newaxis]
        nbrs_delta = _dist(
            targets_pos, nbrs_delta[..., jnp.newaxis, :], periodicity=self._periodicity
        )
        return self._kernel(targets_pos, nbrs_delta)

    def get_matrices(self, index, level):
        # FIXME shizophrenic behavior of level
        (targets, _), ((nbrs, _),) = self.get_output_input_indices(index, level)
        return (self.evaluate_kernel(targets, nbrs, level + 1),)


class IsoPowerInterpolationKernel(KernelBase):
    def __init__(
        self, grid: Grid, window_size: Iterable[int], scale=None, p=2, periodicity=None
    ):
        self._scale = scale
        self._p = p
        self._window_size = window_size
        self._periodicity = periodicity
        super().__init__(grid=grid)

    def get_output_input_indices(self, index, level):
        if level == self.grid.depth:
            raise ValueError("Finest level has no children to interpolate to")
        g = self.grid.at(level)
        target = g.children(index).reshape(index.shape + (-1,))
        source = g.neighborhood(index, self._window_size)
        source = source.reshape(index.shape + (-1,))
        return (target, level + 1), ((source, level),)

    def get_matrices(self, index, level):
        """Get distances between neighbors and distances to children from an index."""
        target, source = self.get_output_input_indices(index, level)
        coord_source = self.grid.at(level).index2coord(source)
        coord_target = self.grid.at(level + 1).index2coord(target)

        child_dist = _dist(
            coord_target[..., jnp.newaxis],
            coord_source[..., jnp.newaxis, :],
            self._periodicity,
        )
        distances = _dist(
            coord_source[..., jnp.newaxis],
            coord_source[..., jnp.newaxis, :],
            self._periodicity,
        )
        child_dist = jnp.linalg.norm(child_dist, axis=0)
        distances = jnp.linalg.norm(distances, axis=0)

        if self._scale is None:
            scale = np.mean(distances[distances != 0])
        else:
            scale = self._scale

        def _ker(dist):
            dist = jnp.abs(dist) / scale
            res = jnp.exp(-dist)
            if self._p == 0:
                return res
            if self._p == 1:
                dist *= np.sqrt(3)
                return (1.0 + dist) * res
            if self._p == 2:
                dist *= np.sqrt(5)
                return (1.0 + dist + dist**2 / 3.0) * res
            raise NotImplementedError(f"No interpolation method for p={self._p} found.")

        wgts = _ker(child_dist) @ jnp.linalg.inv(_ker(distances))
        shp = wgts.shape[-2:]
        pre = np.eye(shp[1]) - np.ones((shp[1],) * 2) / shp[1]
        post = np.ones(shp) / shp[1]
        sl = (np.newaxis,) * (wgts.ndim - 2) + (slice(None),) * 2
        return (wgts @ pre[sl] + post[sl],)


class CombineKernel(KernelBase):
    def get_output_input_indices(self, index, level):
        if level == self.grid.depth:
            raise ValueError("Finest level has no children to combine")
        return (index, level), (
            (
                self.grid.at(level).children(index).reshape(index.shape + (-1,)),
                level + 1,
            ),
        )

    def get_matrices(self, index, level):
        _, ((children, _),) = self.get_output_input_indices(index, level)
        return (jnp.ones(children.shape[1:-1] + (1, children.shape[-1])),)


class MSCRefine(KernelBase):
    def __init__(
        self,
        interpolation: KernelBase,
        convolution: FixedKernelFunctionBatch,
        combine: KernelBase,
    ):
        self._interpolation = interpolation
        self._convolution = convolution
        self._combine = combine
        assert self._convolution.grid == self._interpolation.grid
        super().__init__(grid=self._convolution.grid)

    def get_output_input_indices(self, index, level):
        target, cindex = self._convolution.get_output_input_indices(index, level)
        if level == 0:
            ids = ()
        else:
            itarget, iindex = self._interpolation.get_output_input_indices(
                index, level - 1
            )
            assert itarget == target
            ids = ((iindex, level - 1),)
        return (target, level), ids + ((cindex, level),)

    def get_matrices(self, index, level):
        # FIXME shizophrenic behavior of level
        ker = self._convolution.get_matrices(index, level)
        if level == 0:
            return (ker,)
        iwgts = self._interpolation.get_matrices(index, level)

        _, iindex, cindex = self.get_output_input_indices(index, level)

        parent_kernel = self._convolution.evaluate_kernel(iindex, iindex, level)
        assert parent_kernel.shape[:-2] == iwgts.shape[:-2]

        cparent, children = self._combine.get_output_input_indices(iindex, level)
        assert np.all(cparent == iindex)
        combine = self._combine.get_matrices(iindex, level)
        combine = combine[..., 0, :]  # Remove single axis for output
        assert combine.shape[:-2] == iwgts.shape[:-2]
        # Rearange matrices `combine` to map from convolution nbrs to coarse points
        ncoarse = combine.shape[-2]
        nbase = combine.shape[-1]
        tmp = jnp.zeros(combine.shape[:-1] + (ncoarse * nbase,))
        for i in range(ncoarse):
            sl = (slice(None),) * (combine.ndim - 2) + (
                i,
                slice(nbase * i, nbase * (i + 1)),
            )
            tmp = tmp.at[sl].set(combine[..., i, :])
        children = children.reshape(children.shape[:-2] + (-1, 1))
        isin = children == cindex[..., jnp.newaxis, :]
        isin = jnp.all(isin, axis=0)
        isin = jnp.any(isin, axis=-1)

        def f(ar, inid):
            inid = jnp.where(inid, size=cindex.shape[-1])[0]
            return ar[:, inid]

        for _ in range(combine.ndim - 2):
            f = vmap(f)
        combine = f(tmp, isin)

        parent_kernel = iwgts @ parent_kernel @ combine
        return (iwgts, ker - parent_kernel)
