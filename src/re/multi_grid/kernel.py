# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Philipp Frank
from dataclasses import dataclass, field
from functools import partial, reduce
from typing import Callable, Iterable, Optional, Tuple, Union
from jax import vmap, jit, eval_shape
from jax.tree_util import tree_map
from jax.lax import scan
from .indexing import Grid, FlatGrid
from ..num import amend_unique_
from ..tree_math.vector_math import ShapeWithDtype
from ..refine.util import refinement_matrices
from ..model import Model

import numpy as np
import jax.numpy as jnp
import numpy.typing as npt


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


class KernelBase(Model):
    grid: Grid = field(metadata=dict(static=True))  # TODO
    eval_indices: Tuple[npt.NDArray[np.int_]] = field(metadata=dict(static=False))

    def __init__(self, grid, *, dtype=jnp.float_):
        if not isinstance(grid, Grid):
            raise ValueError(f"Unknown grid type: {grid}")
        self.grid = grid

        eval_indices = []
        domain = []
        for lvl in range(self.grid.depth + 1):
            atLevel = self.grid.at(lvl)
            if lvl != self.grid.depth:
                eval_indices.append(atLevel.refined_indices())
            domain.append(ShapeWithDtype(atLevel.shape, dtype=dtype))
        self.eval_indices = tuple(eval_indices)
        super().__init__(domain=domain, white_init=True)

    def __getitem__(self, key):
        return self.get_kernel(key[0], key[1])

    def get_indices(self, index, level):
        raise NotImplementedError

    def get_kernel(self, index, level):
        raise NotImplementedError

    def get_distance_kernel(self, index, level, norm=partial(jnp.linalg.norm, axis=0)):
        if level == -1:
            raise NotImplementedError
        (out, olvl), ids = self.get_indices(index, level)
        out = out.reshape(index.shape + (-1,))
        out = self.grid.at(olvl).index2coord(out)
        assert index.ndim == out.ndim - 1
        ids = tuple(self.grid.at(ii[1]).index2coord(ii[0]) for ii in ids)
        ids = jnp.concatenate(ids, axis=-1)
        assert index.ndim == ids.ndim - 1
        return (norm(out[..., jnp.newaxis] - ids[..., jnp.newaxis, :]),)

    def _freeze(
        self,
        *,
        rtol=1e-5,
        atol=1e-10,
        buffer_size=10000,
        use_distances=True,
    ):
        """Evaluate the kernel and store it in a `FrozenKernel`. Kernels may be
        grouped by similarity and accessed via lookup"""
        fgrid = self.grid if isinstance(self.grid, FlatGrid) else FlatGrid(self.grid)
        uindices = []
        invindices = []
        indexmaps = []
        for lvl in range(self.grid.depth):
            atlevel = self.grid.at(lvl)
            fatlevel = fgrid.at(lvl)

            def get_kernel(id):
                id = jnp.atleast_1d(id)
                f = self.get_distance_kernel if use_distances else self.get_kernel
                ker = f(fatlevel.flatindex2index(id), lvl)
                ker = jnp.concatenate(tuple(kk.ravel() for kk in ker))
                return ker

            @jit
            def scanned_amend_unique(carry, idx, shift):
                (u, inv) = carry
                k = get_kernel(idx)
                u, invid = amend_unique_(u, k, axis=0, atol=atol, rtol=rtol)
                inv = inv.at[idx - shift].set(invid)
                return (u, inv), invid

            indices = atlevel.refined_indices()
            indices = fatlevel.index2flatindex(indices)[0].ravel()
            shift = np.min(indices)
            size = np.max(indices) - shift
            myinv = np.full(size, buffer_size + 1)

            shp = eval_shape(get_kernel, indices[0]).shape
            unique = jnp.full((buffer_size,) + shp, jnp.nan)

            (unique, myinv), inv = scan(
                partial(scanned_amend_unique, shift=shift), (unique, myinv), indices
            )
            _, idx = np.unique(inv, return_index=True)
            n = idx.size
            if n >= unique.shape[0] or not np.all(np.isnan(unique[n:])):
                raise ValueError("`mat_buffer_size` too small")
            uids = indices[idx]
            uids = fatlevel.flatindex2index(uids[np.newaxis, :])
            uindices.append(uids)
            invindices.append(myinv)
            indexmaps.append(_IdxMap(shift, fatlevel.index2flatindex))
        return uindices, invindices, indexmaps

    def freeze(
        self,
        *,
        uindices=None,
        invindices=None,
        indexmaps=None,
        rtol=1e-5,
        atol=1e-10,
        buffer_size=10000,
        use_distances=True,
    ):
        """Evaluate the kernel and store it in a `FrozenKernel`. Kernels may be
        grouped by similarity and accessed via lookup"""
        if uindices is None:
            uindices, invindices, indexmaps = self._freeze(
                rtol=rtol,
                atol=atol,
                buffer_size=buffer_size,
                use_distances=use_distances,
            )
        elif (indexmaps is None) or (invindices is None):
            raise ValueError(
                "`uindices`,  `invindices`, and `indexmaps` must be either both None or not None"
            )
        return _FrozenKernel(self, uindices, invindices, indexmaps)

    def apply_at(self, index, level, x, **kwargs):
        assert index.ndim == 1
        iout, iin = self.get_indices(index, level, **kwargs)
        kernels = self.get_kernel(index, level, **kwargs)
        assert len(iin) == len(kernels)
        res = reduce(
            lambda a, b: a + b,
            (kk @ x[ii[1]][tuple(ii[0])] for kk, ii in zip(kernels, iin)),
        )
        return iout, res.reshape(iout[0].shape[1:])

    def apply(self, x, *, copy=True, _eval_indices=None, **kwargs):
        """Applies the kernel to values on an entire multigrid"""
        xd, gd = len(x), self.grid.depth
        if xd != (gd + 1):
            raise ValueError(f"Input of length {xd} does not match grid depth {gd}")
        for lvl, xx in enumerate(x):
            xs, gs = xx.shape, self.grid.at(lvl).shape
            if np.any(xs != gs):
                msg = f"Input at level {lvl} of shape {xs} does not match grid of shape {gs}"
                raise ValueError(msg)

        x = list(jnp.copy(xx) for xx in x) if copy else x
        # Use dummy index for base
        _, x[0] = self.apply_at(jnp.atleast_1d(-1), -1, x, **kwargs)
        eval_indices = self.eval_indices if _eval_indices is None else _eval_indices
        for lvl, index in enumerate(eval_indices):
            atlevel = self.grid.at(lvl)
            # TODO this selects window for each index individually
            f = partial(self.apply_at, **kwargs)
            ndim = atlevel.ndim
            for i in range(ndim):
                f = vmap(f, (1, None, None), ((ndim - i, None), ndim - i - 1))
            (_, level), res = f(index, lvl, x)
            x[level] = self.grid.at(level).resort(res)
        return x

    def __call__(self, x):
        return self.apply(x)


# TODO replace
@dataclass()
class _IdxMap:
    shift: int
    index2flatindex: callable


def _eval_kernel(func, indices, level, batchsize=1024):
    res = []
    for i in range(1 + (indices.shape[1] // batchsize)):
        res.append(func(indices[:, i * batchsize : (i + 1) * batchsize], level))
    return tree_map(lambda *args: jnp.concatenate(args, axis=0), *res)


class _FrozenKernel(KernelBase):
    def __init__(self, kernel: KernelBase, uindices, invindices, indexmaps):
        self.get_indices = kernel.get_indices
        self._invindices = invindices
        self._indexmaps = indexmaps
        self._kernels = tuple(
            kernel.get_kernel(ii, ll)
            for ll, ii in enumerate(uindices)
            # _eval_kernel(kernel.get_kernel, ii, ll) FIXME
            for ll, ii in enumerate(uindices)
        )
        self._base_kernel = kernel.get_kernel(jnp.atleast_1d(-1), -1)
        super().__init__(grid=kernel.grid)

    def get_kernel(self, index, level):
        if level == -1:
            return self._base_kernel

        map_atlevel = self._indexmaps[level]
        inv_atlevel = self._invindices[level]
        index = map_atlevel.index2flatindex(index)[0]
        index = inv_atlevel[index - map_atlevel.shift]
        return tuple(kk[index] for kk in self._kernels[level])


class ICRefine(KernelBase):
    def __init__(
        self,
        grid: Grid,
        window_size: Union[Iterable, int],
        covariance: Optional[Callable] = None,
    ):
        self._covariance = covariance
        window_size = (window_size,) if isinstance(window_size, int) else window_size
        self._window_size = tuple(window_size)
        reset = False
        if self._covariance is None:
            self._covariance = lambda x,y: jnp.linalg.norm(x-y)
            reset = True
        super().__init__(grid=grid)
        if reset:
            self._covariance = None

    def get_indices(self, index, level, **kwargs):
        if level == -1:
            grid_at_lvl = self.grid.at(0)
            pixel_indices = np.mgrid[tuple(slice(0, sz) for sz in grid_at_lvl.shape)]
            assert pixel_indices.shape[0] == grid_at_lvl.ndim
            return (pixel_indices, 0), (
                (pixel_indices.reshape(grid_at_lvl.ndim, -1), 0),
            )
        if (level >= self.grid.depth) or (level < -1):
            mg = f"Level {level} out of bounds for grid deph {self.grid.depth}"
            raise ValueError(mg)
        atlevel = self.grid.at(level)
        assert index.shape[0] == atlevel.ndim
        gc = atlevel.neighborhood(index, self._window_size).reshape(index.shape + (-1,))
        gout = self.grid.at(level).children(index)
        gf = gout.reshape(index.shape + (-1,))
        return (gout, level + 1), ((gc, level), (gf, level + 1))

    def get_kernel(self, index, level, _covariance=None):
        fcov = self._covariance if _covariance is None else _covariance
        fcov = vmap(fcov, in_axes=(None, -1), out_axes=-1)
        fcov = vmap(fcov, in_axes=(-1, None), out_axes=-1)

        if level == -1:
            _, ((ids, _),) = self.get_indices(index, -1)
            gc = self.grid.at(0).index2coord(ids)
            assert gc.ndim == 2
            cov = fcov(gc, gc)
            assert cov.shape == (gc.shape[1],) * 2
            from ..refine.util import projection_MatrixSq

            return (projection_MatrixSq(cov),)
        _, ((idc, _), (idf, _)) = self.get_indices(index, level)

        def _get_kernel(gc, gf):
            gc = self.grid.at(level).index2coord(gc)
            gf = self.grid.at(level + 1).index2coord(gf)
            assert gc.shape[0] == gf.shape[0]
            assert gc.ndim == gf.ndim == 2
            coord = jnp.concatenate((gc, gf), axis=-1)
            cov = fcov(coord, coord)
            return refinement_matrices(cov, gf.shape[1])

        f = _get_kernel
        for _ in range(index.ndim - 1):
            f = vmap(f, in_axes=(1, 1))
        return f(idc, idf)


# FIXME old code from here
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
