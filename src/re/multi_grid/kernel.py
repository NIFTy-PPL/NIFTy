# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Philipp Frank
from functools import partial, reduce
import operator
from typing import Callable, Iterable
from jax.tree_util import register_pytree_node_class
from jax import vmap, jit
from jax.experimental.checkify import checkify
from .indexing import Grid

import numpy as np
import jax.numpy as jnp


def _mydist(x, y, periodicity=None):
    if periodicity is not None:
        L = np.atleast_1d(periodicity)
        d = y - x
        L = L[(slice(None),) + (np.newaxis,) * (d.ndim - 1)]
        return jnp.where(jnp.abs(d) > L / 2.0, d - jnp.sign(d) * L, d)
    return x - y


"""
Ideas for kernel:
- A structure on a multigrid that can apply linear operators with inputs accross
  the grid
- A Kernel could be as simple as coarse graining children to a parent; or could
  be a full ICRefine or MSCrefine
- Should be fully jax transformable (TODO) to allow seamlessly being used in
  larger models (e.g. models for variable kernels should be able to output an
  instance of 'kernel')
- Unified way of common functions such as `apply_at`, `apply`, `freeze` (see
  below)
- Current internal logic: `index` always refers to a parent and operations (such
  as refine/convolve) are constructed for all children of this parent.
"""


@register_pytree_node_class
class KernelBase:
    def __getitem__(self, key):
        raise NotImplementedError

    @property
    def grid(self):
        return self._grid

    def get_indices(self, index, level):
        raise NotImplementedError

    def get_kernel(self, index, level):
        raise NotImplementedError

    def freeze(self, *, indices=None, indexmaps=None, rtol=1e-5, atol=1e-5, nbatch=10):
        """Evaluate the kernel and store it in a `FrozenKernel`. Kernels may be
        grouped by similarity and accessed via lookup"""
        if (indices is None) and (indexmaps is None):
            raise NotImplementedError  # TODO
            indices = []
            indexmaps = []
            for lvl in range(self.grid.depth):
                index = np.mgrid[
                    tuple(slice(0, sh) for sh in range(self.grid.at(lvl).shape))
                ]
                index = index.reshape((index.shape[0], -1))
                if index.shape[1] > nbatch:
                    bsz = index.shape[1] // nbatch
                    index = tuple(
                        index[:, bsz * i : min(bsz * (i + 1), index.shape[1])]
                        for i in range(nbatch)
                    )
                else:
                    index = (index,)
                kernels = None
                getker = jit(self.get_kernel, static_argnums=1)
                for ii in index:
                    kers = getker(ii, lvl)
        elif indexmaps is None:
            raise ValueError(
                "`indices` and `indexmaps` must be either both None or not None"
            )
        return FrozenKernel(self, indexmaps, indices)

    def apply_at(self, index, level, x):
        assert index.ndim == 1
        iout, iin = self.get_indices(index, level)
        kernels = self.get_kernel(index, level)
        assert len(iin) == len(kernels)
        return iout, reduce(
            lambda a, b: a + b,
            (kk @ x[ii[1]][tuple(ii[0])] for kk, ii in zip(kernels, iin)),
        )

    def apply(self, x, indices=None, copy=True):
        """Applies the kernel to values on an entire multigrid"""
        xd, gd = len(x), self.grid.depth
        if xd != (gd + 1):
            raise ValueError(f"Input of length {xd} does not match grid depth {gd}")
        for lvl, xx in enumerate(x):
            xs, gs = xx.size, self.grid.at(lvl).size
            if xs != gs:
                msg = f"Input at level {lvl} of size {xs} does not match grid of size {gs}"
                raise ValueError(msg)

        x = list(jnp.copy(xx) for xx in x) if copy else x
        # Use dummy index for base
        _, x[0] = self.apply_at(jnp.atleast_1d(-1), -1, x)
        for lvl in range(self.grid.depth):
            if indices is None:
                index = self.grid.at(lvl).refined_indices()
            else:
                index = indices[lvl]
            # TODO this selects window for each index individually
            f = self.apply_at
            for i in range(index.ndim - 1):
                f = vmap(f, (1, None, None), ((1, None), i))
            (_, level), res = f(index, lvl, x)
            x[level] = res.reshape(x[level].shape)
        return x

    def tree_flatten(self):
        raise NotImplementedError
        # return ((self.x, self.y), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        raise NotImplementedError
        # return cls(*children)


class FrozenKernel(KernelBase):
    def __init__(self, kernel: KernelBase, indexmaps, indices):
        self.get_indices = kernel.get_indices
        self._indexmaps = indexmaps
        self._kernels = tuple(
            kernel.get_kernel(ii, ll) for ll, ii in enumerate(indices)
        )

    def get_kernel(self, index, level):
        return self._kernels[level][self._indexmaps[level][index]]


class FixedKernelFunctionBatch(KernelBase):
    def __init__(
        self, grid: Grid, window_size: Iterable[int], kernel: Callable, periodicity=None
    ):
        self._grid = grid
        self._kernel = kernel
        self._window_size = window_size  # TODO different at different levels?
        self._periodicity = periodicity

    def get_indices(self, index, level):
        if level == 0:
            grid = np.mgrid[tuple(slice(0, s) for s in self.grid.shape)]
            grid = grid.reshape((grid.shape[0], -1))
            assert grid.shape[0] == index.shape[0]

            return (
                index[(slice(None),) * index.ndim + (1,)]
                * grid[(slice(None),) + (1,) * index.ndim + (slice(None),)]
            )
        targets = self._grid.at(level - 1).children(index).reshape(index.shape + (-1,))
        nbrs = (
            self._grid.at(level)
            .neighborhood(targets[..., -1], self._window_size)
            .reshape(index.shape + (-1,))
        )
        return (targets, level), ((nbrs, level),)

    def evaluate_kernel(self, out_index, in_index, level):
        gridAtLevel = self.grid.at(level)
        targets_pos = gridAtLevel.index2coord(out_index)
        nbrs_delta = gridAtLevel.index2coord(in_index)
        targets_pos = targets_pos[..., jnp.newaxis]
        nbrs_delta = _mydist(
            targets_pos, nbrs_delta[..., jnp.newaxis, :], periodicity=self._periodicity
        )
        return self._kernel(targets_pos, nbrs_delta)

    def get_kernel(self, index, level):
        # FIXME shizophrenic behavior of level
        (targets, _), ((nbrs, _),) = self.get_indices(index, level)
        return (self.evaluate_kernel(targets, nbrs, level + 1),)


class IsoPowerInterpolationKernel(KernelBase):
    def __init__(
        self, grid: Grid, window_size: Iterable[int], scale=None, p=2, periodicity=None
    ):
        self._grid = grid
        self._scale = scale
        self._p = p
        self._window_size = window_size
        self._periodicity = periodicity

    def get_indices(self, index, level):
        if level == self._grid.depth:
            raise ValueError("Finest level has no children to interpolate to")
        gridAtLevel = self._grid.at(level)
        target = gridAtLevel.children(index).reshape(index.shape + (-1,))
        source = gridAtLevel.neighborhood(index, self._window_size)
        source = source.reshape(index.shape + (-1,))
        return (target, level + 1), ((source, level),)

    def get_kernel(self, index, level):
        """Get distances between neighbors and distances to children from an index."""
        target, source = self.get_indices(index, level)
        coord_source = self._grid.at(level).index2coord(source)
        coord_target = self._grid.at(level + 1).index2coord(target)

        child_dist = _mydist(
            coord_target[..., jnp.newaxis],
            coord_source[..., jnp.newaxis, :],
            self._periodicity,
        )
        distances = _mydist(
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
    def __init__(self, grid):
        self._grid = grid

    def get_indices(self, index, level):
        if level == self._grid.depth:
            raise ValueError("Finest level has no children to combine")
        return (index, level), (
            (
                self._grid.at(level).children(index).reshape(index.shape + (-1,)),
                level + 1,
            ),
        )

    def get_kernel(self, index, level):
        _, ((children, _),) = self.get_indices(index, level)
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
        self._grid = self._convolution.grid

    def get_indices(self, index, level):
        target, cindex = self._convolution.get_indices(index, level)
        if level == 0:
            ids = ()
        else:
            itarget, iindex = self._interpolation.get_indices(index, level - 1)
            assert itarget == target
            ids = ((iindex, level - 1),)
        return (target, level), ids + ((cindex, level),)

    def get_kernel(self, index, level):
        # FIXME shizophrenic behavior of level
        ker = self._convolution.get_kernel(index, level)
        if level == 0:
            return (ker,)
        iwgts = self._interpolation.get_kernel(index, level)

        _, iindex, cindex = self.get_indices(index, level)

        parent_kernel = self._convolution.evaluate_kernel(iindex, iindex, level)
        assert parent_kernel.shape[:-2] == iwgts.shape[:-2]

        cparent, children = self._combine.get_indices(iindex, level)
        assert np.all(cparent == iindex)
        combine = self._combine.get_kernel(iindex, level)
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


class ICRefine(KernelBase):
    def __init__(self, grid, covariance, window_size):
        self._grid = grid
        mapped_kernel = vmap(covariance, in_axes=(None, -1), out_axes=-1)
        mapped_kernel = vmap(mapped_kernel, in_axes=(-1, None), out_axes=-1)
        self._covariance = mapped_kernel
        window_size = (window_size,) if isinstance(window_size, int) else window_size
        self._window_size = tuple(window_size)

    def get_indices(self, index, level):
        # FIXME inconsistent behavior of level and index
        if level == -1:
            grid_at_lvl = self._grid.at(0)
            pixel_indices = np.mgrid[tuple(slice(0, sz) for sz in grid_at_lvl.shape)]
            assert pixel_indices.shape[0] == grid_at_lvl.ndim
            pixel_indices = pixel_indices.reshape(grid_at_lvl.ndim, -1)
            return (pixel_indices, 0), ((pixel_indices, 0),)
        if (level >= self._grid.depth) or (level < -1):
            mg = f"Level {level} out of bounds for grid deph {self._grid.depth}"
            raise ValueError(mg)
        atlevel = self._grid.at(level)
        assert index.shape[0] == atlevel.ndim
        gc = atlevel.neighborhood(index, self._window_size).reshape(index.shape + (-1,))
        gf = self._grid.at(level).children(index).reshape(index.shape + (-1,))
        return (gf, level + 1), ((gc, level), (gf, level + 1))

    def get_kernel(self, index, level):
        if level == -1:
            _, ((ids, _),) = self.get_indices(index, -1)
            gc = self._grid.at(0).index2coord(ids)
            assert gc.ndim == 2
            cov = self._covariance(gc, gc)
            assert cov.shape == (gc.shape[1],) * 2
            from ..refine.util import projection_MatrixSq

            return (projection_MatrixSq(cov),)
        _, ((idc, _), (idf, _)) = self.get_indices(index, level)

        from ..refine.util import refinement_matrices

        def _get_kernel(gc, gf):
            gc = self.grid.at(level).index2coord(gc)
            gf = self.grid.at(level + 1).index2coord(gf)
            assert gc.shape[0] == gf.shape[0]
            assert gc.ndim == gf.ndim == 2
            coord = jnp.concatenate((gc, gf), axis=-1)
            # del gc, gf
            cov = self._covariance(coord, coord)
            # del coord
            return refinement_matrices(cov, gf.shape[1])

        f = _get_kernel
        for _ in range(index.ndim - 1):
            f = vmap(f, in_axes=(1, 1))
        return f(idc, idf)
