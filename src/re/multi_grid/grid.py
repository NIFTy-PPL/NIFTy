#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Gordian Edenhofer, Philipp Frank

import operator
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Iterable, Optional

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import ShapeDtypeStruct, eval_shape
from jax.experimental import checkify
from jax.lax import select


@dataclass()
class GridAtLevel:
    shape: npt.NDArray[np.int_]
    splits: Optional[npt.NDArray[np.int_]]
    parent_splits: Optional[npt.NDArray[np.int_]]

    def __init__(self, shape, splits=None, parent_splits=None):
        self.shape = np.atleast_1d(shape)
        if splits is not None:
            splits = np.broadcast_to(splits, (self.ndim,))
        if parent_splits is not None:
            parent_splits = np.broadcast_to(parent_splits, (self.ndim,))
        self.splits = splits
        self.parent_splits = parent_splits

    def _parse_index(self, index):
        index = jnp.asarray(index)
        if index.shape[0] != self.shape.size:
            ve = f"index {index} is of invalid length {index.shape[0]} for shape {self.shape}"
            raise IndexError(ve)
        # Follow jax-array style out of bounds handling
        shp_bc = self.shape[(slice(None),) + (np.newaxis,) * (index.ndim - 1)]
        index = select(jnp.abs(index) < shp_bc, index, jnp.sign(index) * (shp_bc - 1))
        return index % shp_bc

    @property
    def size(self):
        return reduce(operator.mul, self.shape, 1)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def raw_grids(self):
        return (self,)

    def refined_indices(self):
        if self.splits is None:
            raise IndexError("this level has no children")
        # TODO non-dense grid
        return np.mgrid[tuple(slice(0, sh) for sh in self.shape)]

    def resort(self, batched_ar, /):
        if batched_ar.ndim != 2 * self.ndim:
            raise ValueError
        if batched_ar.shape[1::2] != tuple(self.parent_splits):
            raise ValueError
        shp = batched_ar.shape
        return batched_ar.reshape(tuple(a * b for a, b in zip(shp[::2], shp[1::2])))

    def children(self, index):
        if self.splits is None:
            raise IndexError("this level has no children")
        index = self._parse_index(index)
        dtp = np.result_type(index)
        f = self.splits[(slice(None),) + (np.newaxis,) * (index.ndim - 1)]
        c = np.mgrid[tuple(slice(sz) for sz in self.splits)].astype(dtp)
        c_bc = (
            (slice(None),)
            + (np.newaxis,) * (index.ndim - 1)
            + (slice(None),) * self.ndim
        )
        ids = index * f
        return ids[(slice(None),) * ids.ndim + (np.newaxis,) * self.ndim] + c[c_bc]

    def neighborhood(self, index, window_size: Iterable[int]):
        index = self._parse_index(index)
        dtp = np.result_type(index)
        window_size = np.asarray(window_size)
        assert window_size.size == self.ndim
        c = np.mgrid[tuple(slice(sz) for sz in window_size)]
        c -= (window_size // 2)[(slice(None),) + (np.newaxis,) * self.ndim]
        c_bc = (
            (slice(None),)
            + (np.newaxis,) * (index.ndim - 1)
            + (slice(None),) * self.ndim
        )
        m_bc = (slice(None),) + (np.newaxis,) * (index.ndim - 1 + self.ndim)
        id_bc = (slice(None),) * index.ndim + (np.newaxis,) * self.ndim
        res = (index[id_bc] + c[c_bc]) % self.shape[m_bc]
        return res.astype(dtp)

    def parent(self, index):
        if self.parent_splits is None:
            raise IndexError("you are alone in this world")
        index = self._parse_index(index)
        bc = (slice(None),) + (np.newaxis,) * (index.ndim - 1)
        return index // self.parent_splits[bc]

    def index2coord(self, index):
        slc = (slice(None),) + (np.newaxis,) * (index.ndim - 1)
        return (index + 0.5) / self.shape[slc]

    def coord2index(self, coord, dtype=np.uint64):
        slc = (slice(None),) + (np.newaxis,) * (coord.ndim - 1)
        index = (coord * self.shape[slc] - 0.5)
        if np.issubdtype(dtype, np.integer):
            return np.rint(index).astype(dtype)
        else:
            raise ValueError(f"non-integer index dtype: {dtype}")

    def index2volume(self, index):
        return np.array(1.0 / self.size)[(np.newaxis,) * index.ndim]


@dataclass()
class Grid:
    """Dense grid with periodic boundary conditions."""

    shape0: npt.NDArray[np.int_]
    splits: tuple[npt.NDArray[np.int_]]
    atLevel: Callable

    def __init__(self, *, shape0, splits, atLevel=GridAtLevel):
        self.shape0 = np.atleast_1d(shape0)
        splits = (splits,) if isinstance(splits, int) else splits
        self.splits = tuple(np.broadcast_to(s, self.shape0.shape) for s in splits)
        self.atLevel = atLevel

    @property
    def depth(self):
        return len(self.splits)

    def _parse_level(self, level):
        if np.abs(level) > self.depth:
            raise IndexError(f"{self.__class__.__name__} does not have level {level}")
        return level % (self.depth + 1)

    def amend(self, splits):
        splits = (splits,) if isinstance(splits, int) else splits
        splits = tuple(np.broadcast_to(s, self.shape0.shape) for s in splits)
        return self.__class__(
            shape0=self.shape0, splits=self.splits + splits, atLevel=self.atLevel
        )

    def at(self, level: int):
        level = self._parse_level(level)
        fct = 1
        if level > 0:
            fct = np.array(
                [reduce(operator.mul, si) for si in zip(*self.splits[:level])]
            )
        return self.atLevel(
            shape=self.shape0 * fct,
            splits=self.splits[level] if level < self.depth else None,
            parent_splits=self.splits[level - 1] if level >= 1 else None,
        )


@dataclass()
class OpenGridAtLevel(GridAtLevel):
    padding: Optional[npt.NDArray[np.int_]]
    parent_padding: Optional[npt.NDArray[np.int_]]
    shifts: Optional[npt.NDArray[np.int_]]

    def __init__(
        self,
        shape,
        splits=None,
        parent_splits=None,
        *,
        padding=None,
        parent_padding=None,
        shifts=None,
        level=None,
        all_splits=None,
    ):
        super().__init__(shape=shape, splits=splits, parent_splits=parent_splits)
        if padding is not None:
            padding = np.broadcast_to(padding, (self.ndim,))
        if parent_padding is not None:
            parent_padding = np.broadcast_to(parent_padding, (self.ndim,))
        if shifts is not None:
            shifts = np.broadcast_to(shifts, (self.ndim,))
        self.padding = padding
        self.parent_padding = parent_padding
        self.shifts = shifts
        del level, all_splits  # not used here but used by child classes

    def refined_indices(self):
        if self.splits is None:
            raise IndexError("ths level has no children")
        # TODO non-dense grid
        return np.mgrid[
            tuple(slice(pp, sh - pp) for sh, pp in zip(self.shape, self.padding))
        ]

    def children(self, index):
        if (self.splits is None) or (self.padding is None):
            raise IndexError("this level has no children")
        lo = self.padding[(slice(None),) + (np.newaxis,) * (index.ndim - 1)]
        hi = self.shape[(slice(None),) + (np.newaxis,) * (index.ndim - 1)] - lo
        # TODO add option for fully jax transformable error handling
        # Follow jax-array inspired out of bounds handling
        return super().children(index.clip(lo, hi - 1) - lo)

    def neighborhood(self, index, window_size: Iterable[int]):
        # TODO add option for fully jax transformable error handling
        # Follow jax-array inspired out of bounds handling
        shp_bc = self.shape[
            (slice(None),) + (np.newaxis,) * (index.ndim - 1 + self.ndim)
        ]
        return super().neighborhood(index, window_size).clip(0, (shp_bc - 1))

    def parent(self, index):
        if self.parent_splits is None:
            raise IndexError("you are alone in this world")
        index = self._parse_index(index)
        bc = (slice(None),) + (np.newaxis,) * (index.ndim - 1)
        return (index // self.parent_splits[bc]) + self.parent_padding[bc]

    def index2coord(self, index):
        slc = (slice(None),) + (np.newaxis,) * (index.ndim - 1)
        shp = self.shape + 2 * self.shifts
        index = index + self.shifts[slc]
        return (index + 0.5) / shp[slc]

    def coord2index(self, coord, dtype=np.uint64):
        slc = (slice(None),) + (np.newaxis,) * (coord.ndim - 1)
        shp = self.shape + 2 * self.shifts
        index = coord * shp[slc] - self.shifts[slc] - 0.5
        if np.issubdtype(dtype, np.integer):
            return np.rint(index).astype(dtype)
        else:
            raise ValueError(f"non-integer index dtype: {dtype}")

    def index2volume(self, index):
        sz = np.prod(self.shape + 2 * self.shifts)
        return np.array(1.0 / sz)[(np.newaxis,) * index.ndim]


@dataclass()
class OpenGrid(Grid):
    """Dense grid with open boundary conditions.

    At each level the grid has all required indices for a full
    refinement/convolution even for those that are not split. However, indices
    used for padding don't have children. The coordinates for the
    refinement/convolution span the full space including any padding on all
    previous layers.
    """

    def __init__(self, *, shape0, splits, padding, atLevel=OpenGridAtLevel):
        super().__init__(shape0=shape0, splits=splits, atLevel=atLevel)
        padding = (padding,) if isinstance(padding, int) else padding
        self.padding = tuple(np.broadcast_to(p, self.shape0.shape) for p in padding)
        if len(self.padding) != len(self.splits):
            msg = f"padding ({padding!r}) and splits ({splits!r}) not of equal length"
            raise ValueError(msg)
        # Validate that the shape is always valid (greater than zero)
        shp = self.shape0
        for si, pd in zip(self.splits, self.padding):
            shp = si * (shp - 2 * pd)
            assert np.all(shp > 0)

    def amend(self, splits, padding):
        splits = (splits,) if isinstance(splits, int) else splits
        splits = tuple(np.broadcast_to(s, self.shape0.shape) for s in splits)
        padding = (padding,) if isinstance(padding, int) else padding
        padding = tuple(np.broadcast_to(p, self.shape0.shape) for p in padding)
        return self.__class__(
            shape0=self.shape0,
            splits=self.splits + splits,
            padding=self.padding + padding,
            atLevel=self.atLevel,
        )

    def at(self, level: int):
        level = self._parse_level(level)
        shp = self.shape0
        shifts = np.zeros_like(shp)
        for si, pd in zip(self.splits[:level], self.padding[:level]):
            shp = si * (shp - 2 * pd)
            shifts = si * (shifts + pd)
        return self.atLevel(
            shape=shp,
            splits=self.splits[level] if level < self.depth else None,
            parent_splits=self.splits[level - 1] if level >= 1 else None,
            padding=self.padding[level] if level < self.depth else None,
            parent_padding=self.padding[level - 1] if level >= 1 else None,
            shifts=shifts,
            level=level,
            all_splits=self.splits[: level + 1],
        )


@dataclass()
class MGridAtLevel(GridAtLevel):
    """Multi-dimensional meshgrid product of multiple grids."""

    grids: tuple[GridAtLevel]

    def __init__(self, *grids):
        self.grids = tuple(grids)

    @property
    def shape(self):
        return np.concatenate(tuple(g.shape for g in self.grids))

    @property
    def splits(self):
        if self.grids[0].splits is None:
            if any(gg.splits is not None for gg in self.grids[1:]):
                raise ValueError(f"inconsistent `None` splits in grids {self.grids}")
            return None
        return np.concatenate(tuple(g.splits for g in self.grids))

    @property
    def parent_splits(self):
        if self.grids[0].parent_splits is None:
            if any(gg.parent_splits is not None for gg in self.grids[1:]):
                msg = f"inconsistent `None` parent_splits in grids {self.grids}"
                raise ValueError(msg)
            return None
        return np.concatenate(tuple(g.parent_splits for g in self.grids))

    @property
    def ngrids(self):
        return len(self.grids)

    @property
    def raw_grids(self):
        return reduce(operator.add, (g.raw_grids for g in self.grids))

    def refined_indices(self):
        mgrids = tuple(gg.refined_indices() for gg in self.grids)
        res = mgrids[0]
        for mg in mgrids[1:]:
            slf = (slice(None),) * res.ndim + (jnp.newaxis,) * (mg.ndim - 1)
            slb = (
                (slice(None),)
                + (jnp.newaxis,) * (res.ndim - 1)
                + (slice(None),) * (mg.ndim - 1)
            )
            shb = res.shape[1:] + mg.shape[1:]
            res = jnp.broadcast_to(res[slf], (res.shape[0],) + shb)
            mg = jnp.broadcast_to(mg[slb], (mg.shape[0],) + shb)
            res = jnp.concatenate((res, mg), axis=0)
        return res

    def resort(self, batched_ar, /):
        if any(isinstance(g, FlatGrid) for g in self.grids):
            raise NotImplementedError()  # TODO generalize using grids resort
        return super().resort(batched_ar)

    def children(self, index) -> npt.NDArray:
        ndims_off = tuple(np.cumsum(tuple(g.ndim for g in self.grids)))
        ndims_sum = ndims_off[-1]
        islice = tuple(slice(l, r) for l, r in zip((0,) + ndims_off[:-1], ndims_off))
        children = tuple(g.children(index[i]) for i, g in zip(islice, self.grids))
        # Make initial entry broadcast-able to the full final shape
        out = children[0][
            (slice(None),) * children[0].ndim
            + (np.newaxis,) * (ndims_sum - self.grids[0].ndim)
        ]
        # Successively concatenate all broadcasted children
        for c, i in zip(children[1:], islice[1:]):
            c = c[
                (slice(None),) * index.ndim
                + (np.newaxis,) * i.start
                + (slice(None),) * (i.stop - i.start)
                + (np.newaxis,) * (ndims_sum - i.stop)
            ]
            assert c.shape[0] == (i.stop - i.start)
            bshp = np.broadcast_shapes(out.shape[1:], c.shape[1:])
            c = jnp.broadcast_to(c, c.shape[:1] + bshp)
            out = jnp.broadcast_to(out, out.shape[:1] + bshp)
            out = jnp.concatenate((out, c), axis=0)
        return out

    def neighborhood(self, index, window_size: tuple[int]):
        ndims_off = tuple(np.cumsum(tuple(g.ndim for g in self.grids)))
        ndims_sum = ndims_off[-1]
        islice = tuple(slice(l, r) for l, r in zip((0,) + ndims_off[:-1], ndims_off))
        window_size = (
            (window_size,) * self.ndim if isinstance(window_size, int) else window_size
        )
        assert len(window_size) == self.ndim
        neighborhood = []
        for i, g in zip(islice, self.grids):
            n = g.neighborhood(index[i], window_size[i])
            neighborhood.append(n)

        # Make initial entry broadcast-able to the full final shape
        out = neighborhood[0][
            (slice(None),) * neighborhood[0].ndim
            + (np.newaxis,) * (ndims_sum - self.grids[0].ndim)
        ]
        # Successively concatenate all broadcasted neighbors
        for n, i in zip(neighborhood[1:], islice[1:]):
            n = n[
                (slice(None),) * index.ndim
                + (np.newaxis,) * i.start
                + (slice(None),) * (i.stop - i.start)
                + (np.newaxis,) * (ndims_sum - i.stop)
            ]
            assert n.shape[0] == (i.stop - i.start)
            bshp = np.broadcast_shapes(out.shape[1:], n.shape[1:])
            n = jnp.broadcast_to(n, n.shape[:1] + bshp)
            out = jnp.broadcast_to(out, out.shape[:1] + bshp)
            out = jnp.concatenate((out, n), axis=0)
        return out

    def parent(self, index):
        ndims_off = tuple(np.cumsum(tuple(g.ndim for g in self.grids)))
        islice = tuple(slice(l, r) for l, r in zip((0,) + ndims_off[:-1], ndims_off))
        parent = tuple(g.parent(index[i]) for i, g in zip(islice, self.grids))
        return jnp.concatenate(parent, axis=0)

    def index2coord(self, index, **kwargs):
        ndims_off = tuple(np.cumsum(tuple(g.ndim for g in self.grids)))
        islice = tuple(slice(l, r) for l, r in zip((0,) + ndims_off[:-1], ndims_off))
        coord = tuple(g.index2coord(index[i]) for i, g in zip(islice, self.grids))
        return jnp.concatenate(coord, axis=0)

    def coord2index(self, coord, **kwargs):
        cdims = tuple(
            eval_shape(gg.index2coord, ShapeDtypeStruct((gg.ndim,), jnp.int_)).shape[0]
            for gg in self.grids
        )
        ndims_off = tuple(np.cumsum(cdims))
        islice = tuple(slice(l, r) for l, r in zip((0,) + ndims_off[:-1], ndims_off))
        index = tuple(g.coord2index(coord[i]) for i, g in zip(islice, self.grids))
        return jnp.concatenate(index, axis=0)

    def index2volume(self, index, **kwargs):
        ndims_off = tuple(np.cumsum(tuple(g.ndim for g in self.grids)))
        islice = tuple(slice(l, r) for l, r in zip((0,) + ndims_off[:-1], ndims_off))
        volume = tuple(g.index2volume(index[i]) for i, g in zip(islice, self.grids))
        return reduce(operator.mul, volume)


@dataclass()
class MGrid(Grid):
    """Multi-dimensional meshgrid product of multiple grids."""

    grids: tuple[Grid]

    def __init__(self, *grids, atLevel=MGridAtLevel):
        self.grids = tuple(grids)
        for i, g in enumerate(grids):
            if not isinstance(g, Grid):
                raise TypeError(
                    f"Grid {g.__class__.__name__} at index {i} of invalid type"
                )
            if g.depth != grids[0].depth:
                raise ValueError(
                    f"Grid {g.__class__.__name__} at index {i} not of same depth"
                )
        shape0 = np.concatenate(tuple(g.shape0 for g in self.grids))
        splits = tuple(
            np.concatenate(tuple(g.splits[lvl] for g in self.grids))
            for lvl in range(self.grids[0].depth)
        )
        super().__init__(shape0=shape0, splits=splits, atLevel=atLevel)

    @property
    def depth(self):
        return self.grids[0].depth

    @property
    def ngrids(self):
        return len(self.grids)

    def amend(self, splits):
        splits = (splits,) if isinstance(splits, int) else splits
        splits = tuple(np.broadcast_to(s, self.shape0.shape) for s in splits)
        # Create slices for indexing individal grid regions in split
        ndims_off = tuple(np.cumsum(tuple(g.ndim for g in self.grids)))
        islice = tuple(slice(l, r) for l, r in zip((0,) + ndims_off[:-1], ndims_off))
        # Separate shared splits into splits for the individual grids
        spg = tuple(zip(*(tuple(spl[i] for i in islice) for spl in splits)))
        grids = tuple(g.amend(s) for g, s in zip(self.grids, spg))
        return self.__class__(*grids, atLevel=self.atLevel)

    def at(self, level: int):
        level = self._parse_level(level)
        return self.atLevel(*tuple(g.at(level) for g in self.grids))


@dataclass()
class FlatGridAtLevel(GridAtLevel):
    """Same as :class:`GridAtLevel` but with a single global integer index for each voxel."""

    grid_at_level: GridAtLevel
    all_shapes: npt.NDArray[np.int_]
    all_splits: tuple[npt.NDArray[np.int_]]
    ordering: str

    def __init__(self, grid_at_level, *, all_shapes, all_splits, ordering="serial"):
        if not isinstance(grid_at_level, GridAtLevel):
            raise TypeError(f"Grid {grid_at_level.__name__} of invalid type")
        self.grid_at_level = grid_at_level
        ordering = str(ordering).lower()
        if ordering not in ("serial", "nest"):
            raise ValueError(f"invalid flat index ordering scheme {ordering}")
        self.ordering = ordering
        self.all_shapes = tuple(np.atleast_1d(sh) for sh in all_shapes)
        self.all_splits = tuple(
            np.broadcast_to(sp, shp.shape)
            for sp, shp in zip(all_splits, self.all_shapes)
        )
        super().__init__(
            shape=(reduce(operator.mul, grid_at_level.shape, 1),),
            splits=None,
            parent_splits=None,
        )

    @property
    def raw_grids(self):
        return self.grid_at_level.raw_grids

    def _weights_serial(self, levelshift):
        if levelshift not in (-1, 0, 1):
            raise ValueError(f"invalid shift in level {levelshift!r}")
        shape = self.all_shapes[-2 + levelshift]
        return np.cumprod(np.append(shape[1:], 1)[::-1])[::-1]

    def _weights_nest(self, levelshift):
        raise NotImplementedError

    def index2flatindex(self, index, levelshift=0):
        # TODO vectorize better
        if self.ordering == "serial":
            wgt = self._weights_serial(levelshift)
            wgt = wgt[(slice(None),) + (np.newaxis,) * (index.ndim - 1)]
            return (wgt * index).sum(axis=0).astype(index.dtype)[jnp.newaxis, ...]
        if self.ordering == "nest":
            raise NotImplementedError
        raise RuntimeError

    def flatindex2index(self, index, levelshift=0):
        # TODO vectorize better
        dtp = index.dtype
        if self.ordering == "serial":
            wgt = self._weights_serial(levelshift)
            tm = jnp.copy(index[0])
            index = np.zeros(wgt.shape + index.shape[1:], dtype=dtp)
            index = []
            for w in wgt:
                tmfl = tm // w
                tm -= w * tmfl
                index.append(tmfl)
            return jnp.stack(index, axis=0).astype(dtp)
        if self.ordering == "nest":
            raise NotImplementedError
        raise RuntimeError

    def refined_indices(self):
        ids = self.grid_at_level.refined_indices()
        return self.index2flatindex(ids).reshape((1, -1))

    def resort(self, batched_ar, /):
        parent_splits = self.all_splits[-3]
        shape = self.all_shapes[-2]
        if batched_ar.ndim != 2:
            raise ValueError
        if batched_ar.shape[1] != np.prod(parent_splits):
            raise ValueError

        if self.ordering == "serial":
            shp = tuple(shape // parent_splits) + tuple(parent_splits)
            batched_ar = batched_ar.reshape(shp)
            ndim = shape.size
            idx = np.arange(ndim)
            axes = reduce(operator.add, ((a, b) for a, b in zip(idx, ndim + idx)))
            return jnp.transpose(batched_ar, axes).ravel()
        if self.ordering == "nest":
            raise NotImplementedError
        raise RuntimeError

    def children(self, index) -> npt.NDArray:
        index = self._parse_index(index)
        index = self.flatindex2index(index)
        children = self.grid_at_level.children(index).reshape(index.shape + (-1,))
        return self.index2flatindex(children, +1)

    def neighborhood(self, index, window_size: Iterable[int]):
        index = self._parse_index(index)
        index = self.flatindex2index(index)
        window = self.grid_at_level.neighborhood(index, window_size=window_size)
        return self.index2flatindex(window.reshape(index.shape + (-1,)))

    def parent(self, index):
        index = self._parse_index(index)
        index = self.flatindex2index(index)
        window = self.grid_at_level.parent(index)
        return self.index2flatindex(window, -1)

    def index2coord(self, index, **kwargs):
        index = self._parse_index(index)
        index = self.flatindex2index(index)
        return self.grid_at_level.index2coord(index, **kwargs)

    def coord2index(self, coord, **kwargs):
        index = self.grid_at_level.coord2index(coord, **kwargs)
        return self.index2flatindex(index)

    def index2volume(self, index, **kwargs):
        index = self._parse_index(index)
        index = self.flatindex2index(index)
        return self.grid_at_level.index2volume(index, **kwargs)


@dataclass()
class FlatGrid(Grid):
    """Same as :class:`Grid` but with a single global integer index for each voxel."""

    grid: Grid
    ordering: str

    def __init__(self, grid, *, ordering="serial", atLevel=FlatGridAtLevel):
        if not isinstance(grid, Grid):
            raise TypeError(f"Grid {grid.__name__} of invalid type")
        self.grid = grid
        ordering = str(ordering).lower()
        if ordering not in ("serial", "nest"):
            raise ValueError(f"invalid flat index ordering scheme {ordering}")
        self.ordering = ordering
        shape0 = np.array([np.prod(grid.shape0)])
        splits = tuple(np.array([np.prod(spl)]) for spl in grid.splits)
        super().__init__(shape0=shape0, splits=splits, atLevel=atLevel)

    def amend(self, splits):
        grid = self.grid.amend(splits)
        return self.__class__(grid=grid, ordering=self.ordering)

    def at(self, level: int):
        level = self._parse_level(level)
        shapes = []
        splits = []
        for lvl in range(level + 2):
            if lvl <= self.depth:
                atlvl = self.grid.at(lvl)
                shapes.append(atlvl.shape)
                splits.append(atlvl.splits)
            else:
                shapes.append(None)
                splits.append(None)
        return self.atLevel(
            self.grid.at(level),
            all_shapes=shapes,
            all_splits=splits,
            ordering=self.ordering,
        )


@dataclass()
class SparseGridAtLevel(FlatGridAtLevel):
    mapping: npt.NDArray[np.int_]
    parent_mapping: Optional[npt.NDArray[np.int_]] = None
    children_mapping: Optional[npt.NDArray[np.int_]] = None

    def __init__(
        self,
        grid_at_level,
        all_shapes,
        all_splits,
        mapping,
        ordering="nest",
        parent_mapping=None,
        children_mapping=None,
    ):
        if not isinstance(grid_at_level, GridAtLevel):
            raise TypeError(f"Grid {grid_at_level.__name__} of invalid type")
        self.mapping = mapping
        self.parent_mapping = parent_mapping
        self.children_mapping = children_mapping
        super().__init__(
            grid_at_level=grid_at_level,
            all_shapes=all_shapes,
            all_splits=all_splits,
            ordering=ordering,
        )
        # Overrides shape to utilize base functions
        self.shape = np.array([self.mapping.size])

    def _mapping(self, levelshift):
        if levelshift == -1:
            mapping = self.parent_mapping
        elif levelshift == 0:
            mapping = self.mapping
        elif levelshift == 1:
            mapping = self.children_mapping
        else:
            raise ValueError(f"invalid shift in level: {levelshift}")
        return mapping

    def arrayindex2flatindex(self, index, levelshift=0):
        index = self._parse_index(index)
        return self._mapping(levelshift)[index]

    def flatindex2arrayindex(self, index, levelshift=0):
        mapping = self._mapping(levelshift)
        arrayid = np.searchsorted(mapping, index)
        #  TODO Benchmark searchsorted on stack instead of second one with `right`
        valid = np.searchsorted(mapping, index, side="right") == arrayid + 1

        checkify.check(
            jnp.all(valid),
            f"flat index {{ids}} not on child grid of {self.__class__.__name__}",
            ids=arrayid[~valid],
        )
        return arrayid

    def refined_indices(self):
        raise NotImplementedError  # TODO

    def children(self, index) -> npt.NDArray:
        index = self.arrayindex2flatindex(index)
        index = self.flatindex2index(index)
        children = self.grid_at_level.children(index)
        children = self.index2flatindex(children, +1)
        return self.flatindex2arrayindex(children, +1)

    def neighborhood(self, index, window_size: Iterable[int]):
        window = self.arrayindex2flatindex(index)
        window = self.flatindex2index(window)
        window = self.grid_at_level.neighborhood(index, window_size=window_size)
        window = self.index2flatindex(window)
        return self.flatindex2arrayindex(window)

    def parent(self, index):
        index = self.arrayindex2flatindex(index)
        index = self.flatindex2index(index)
        parent = self.grid_at_level.parent(index)
        parent = self.index2flatindex(parent, -1)
        return self.flatindex2arrayindex(parent, -1)

    def index2coord(self, index, **kwargs):
        index = self.arrayindex2flatindex(index)
        index = self.flatindex2index(index)
        return self.grid_at_level.index2coord(index, **kwargs)

    def coord2index(self, coord, **kwargs):
        index = self.grid_at_level.coord2index(coord, **kwargs)
        index = self.index2flatindex(index)
        return self.flatindex2arrayindex(index)

    def index2volume(self, index, **kwargs):
        index = self.arrayindex2flatindex(index)
        index = self.flatindex2index(index)
        return self.grid_at_level.index2volume(index, **kwargs)

    def to_flat_grid(self):
        # FIXME @ph-frank incompatible with the current `FlatGridAtLevel` impl
        return FlatGridAtLevel(
            self.grid_at_level,
            self.shape0,
            self.all_parent_splits,
            ordering=self.ordering,
        )


@dataclass()
class SparseGrid(FlatGrid):
    """Same as :class:`FlatGrid` but keeping track of the indices that are actually
    being modeled. This class is especially convenient for open boundary conditions
    but works for arbitrarily sparsely resolved grids."""

    mapping: tuple[npt.NDArray[np.int_]]

    def __init__(
        self,
        grid,
        mapping,
        ordering="nest",
        *,
        _check_mapping=True,
        atLevel=SparseGridAtLevel,
    ):
        if not isinstance(grid, Grid):
            raise TypeError(f"Grid {grid.__class__.__name__} of invalid type")
        self.grid = grid
        self.ordering = ordering

        mapping = (mapping,) if not isinstance(mapping, tuple) else mapping
        mapping = tuple(np.atleast_1d(m) for m in mapping)

        if _check_mapping:
            if len(mapping) != grid.depth + 1:
                md, gd = len(mapping), grid.depth
                nm = grid.__class__.__name__
                msg = f"Map depth {md} does not match grid {nm} depth {gd}"
                raise ValueError(msg)
            for mm in mapping:
                if mm.ndim != 1:
                    raise IndexError("Mapping must be one dimensional")
                if np.any(mm[1:] <= mm[:-1]):
                    raise IndexError("Mapping must be unique and sorted")
        self.mapping = mapping

        super().__init__(grid=grid, ordering=ordering, atLevel=atLevel)

    def amend(self, splits, mapping, **kwargs):
        grid = self.grid.amend(splits, **kwargs)
        mapping = (mapping,) if not isinstance(mapping, tuple) else mapping
        return self.__class__(grid, mapping, nest=self.nest)

    def at(self, level: int):
        level = self._parse_level(level)
        grid_at_level = self.grid.at(level)
        parent_mapping = None if level == 0 else self.mapping[level - 1]
        children_mapping = None if level == self.depth else self.mapping[level + 1]
        return self.atLevel(
            grid_at_level,
            # FIXME @ph-frank pass `all_shapes`,
            self.grid.splits[: (level + 1)] + ((None,) if level == self.depth else ()),
            self.mapping[level],
            ordering=self.ordering,
            parent_mapping=parent_mapping,
            children_mapping=children_mapping,
        )
