#!/usr/bin/env python3

import operator
from dataclasses import dataclass
from functools import partial, reduce
from typing import Callable, Iterable, Optional

import numpy as np
import numpy.typing as npt


@dataclass()
class GridAtLevel:
    shape: npt.NDArray[np.int_]
    splits: npt.NDArray[np.int_]
    parent_splits: Optional[npt.NDArray[np.int_]] = None

    def __init__(self, shape, splits, parent_splits):
        self.shape = np.atleast_1d(shape)
        self.splits = np.atleast_1d(splits)
        self.parent_splits = np.atleast_1d(parent_splits)

    def _parse_index(self, index):
        index = np.asarray(index)
        if np.any(np.any(np.abs(idx) >= s) for idx, s in zip(index, self.shape)):
            nm = self.__class__.__name__
            ve = f"index {index} is out of bounds for {nm} with shape {self.shape}"
            raise IndexError(ve)
        return index % self.shape[(slice(None),) + (np.newaxis,) * (index.ndim - 1)]

    @property
    def size(self):
        return reduce(operator.mul, self.shape, 1)

    @property
    def ndim(self):
        return len(self.shape)

    def children(self, index) -> np.ndarray:
        index = self._parse_index(index)
        dtp = np.result_type(index)
        f = self.splits[(slice(None),) + (np.newaxis,) * (index.ndim - 1)]
        c = np.mgrid[tuple(slice(sz) for sz in self.splits)].astype(dtp)
        c_bc = (
            (slice(None),)
            + (np.newaxis,) * (index.ndim - 1)
            + (slice(None),) * self.ndim
        )
        return (index * f)[..., (np.newaxis,) * self.ndim] + c[c_bc]

    def neighborhood(self, index, window_size: Iterable[int], ensemble_axis=None):
        index = self._parse_index(index)
        dtp = np.result_type(index)
        if ensemble_axis is not None:
            raise NotImplementedError()
        window_size = np.asarray(window_size)
        c = np.mgrid[tuple(slice(sz) for sz in window_size)].astype(dtp)
        c -= (window_size // 2)[:, (np.newaxis,) * self.ndim]
        c_bc = (
            (slice(None),)
            + (np.newaxis,) * (index.ndim - 1)
            + (slice(None),) * self.ndim
        )
        m_bc = (slice(None),) + (np.newaxis,) * (index.ndim - 1 + self.ndim)
        return (index[..., (np.newaxis,) * self.ndim] + c[c_bc]) % self.shape[m_bc]

    def parent(self, index):
        if self.parent_splits is None:
            raise IndexError("you are alone in this world")
        index = self._parse_index(index)
        bc = (slice(None),) + (np.newaxis,) * (index.ndim - 1)
        return index // self.parent_splits[bc]

    def index2coord(self, index, **kwargs):
        return NotImplementedError()

    def coord2index(self, coord, **kwargs):
        return NotImplementedError()

    def index2volume(self, index, **kwargs):
        return NotImplementedError()


@dataclass()
class Grid:
    """Dense grid with periodic boundary conditions.

    Open boundary conditions can be emulated by leaving out indices."""

    shape0: npt.NDArray[np.int_]
    splits: tuple[npt.NDArray[np.int_]]
    atLevel: Callable

    def __init__(self, *, shape0, splits, atLevel=GridAtLevel):
        self.shape0 = np.asarray(shape0)
        splits = (splits,) if isinstance(splits, int) else splits
        self.splits = tuple(np.atleast_1d(s) for s in splits)
        self.atLevel = atLevel

    @property
    def depth(self):
        return len(self.splits)

    def _parse_level(self, level):
        if np.abs(level) >= self.depth:
            raise IndexError(f"{self.__class__.__name__} does not have level {level}")
        return level % self.depth

    def amend(self, splits):
        splits = (splits,) if isinstance(splits, int) else splits
        splits = tuple(np.atleast_1d(s) for s in splits)
        return self.__class__(shape0=self.shape0, splits=self.splits + splits)

    def at(self, level: int):
        level = self._parse_level(level)
        fct = np.array(
            [reduce(operator.mul, si, 1) for si in zip(*self.splits[:level])]
        )
        return self.atLevel(
            shape=self.shape0 * fct,
            splits=self.splits[level],
            parent_splits=self.splits[level - 1] if level >= 1 else None,
        )


class RegularGridAxisAtLevel(GridAtLevel):
    def index2coord(self, index):
        return (index + 0.5) / self.shape[:, (np.newaxis,) * (index.ndim - 1)]

    def coord2index(self, coord):
        return coord * self.shape[:, (np.newaxis,) * (coord.ndim - 1)] - 0.5

    def index2volume(self, index):
        return np.array(1.0 / self.size)[(np.newaxis,) * index.ndim]


def _fill_bad_healpix_neighbors(nside, neighbors, nest: bool = True):
    import warnings

    from healpy.pixelfunc import get_all_neighbours

    idx_w_invalid, nbr_w_invalid = np.nonzero(neighbors == -1)
    if idx_w_invalid.size == 0:
        return neighbors

    # Account for unknown neighbors, encoded by -1
    uniq_idx_w_invalid = np.unique(idx_w_invalid)
    nbr_invalid = neighbors[:, 1:][uniq_idx_w_invalid]
    with warnings.catch_warnings():
        wmsg = "invalid value encountered in _get_neigbors"
        warnings.filterwarnings("ignore", message=wmsg)
        # shape of (n_2nd_neighbors, n_idx_w_invalid, n_1st_neighbors)
        nbr2 = get_all_neighbours(nside, nbr_invalid, nest=nest)
    nbr2 = np.transpose(nbr2, (1, 2, 0))
    nbr2[nbr_invalid == -1] = -1
    nbr2 = nbr2.reshape(uniq_idx_w_invalid.size, -1)
    n_replace = np.sum(neighbors[uniq_idx_w_invalid] == -1, axis=1)
    if np.any(np.diff(n_replace)):
        raise AssertionError()
    n_replace = n_replace[0]
    pix_2nbr = np.stack(
        [
            np.setdiff1d(ar1, ar2)[:n_replace]
            for ar1, ar2 in zip(nbr2, neighbors[uniq_idx_w_invalid])
        ]
    )
    if np.sum(pix_2nbr == -1):
        # `setdiff1d` should remove all `-1` because we worked with rows in
        # neighbors that all contain them
        raise AssertionError()
    # Select a "random" 2nd neighbor to fill in for the missing 1st order
    # neighbor
    neighbors[idx_w_invalid, nbr_w_invalid] = pix_2nbr.ravel()
    if np.sum(neighbors == -1):
        raise AssertionError()
    return neighbors


class HEALPixGridAtLevel(GridAtLevel):
    nside: int
    nest: True
    fill_strategy: str

    def __init__(
        self,
        shape=None,
        splits=4,
        parent_splits=None,
        *,
        nside: int = None,
        nest=True,
        fill_strategy="same",
    ):
        if shape is not None:
            assert nside is None
            assert isinstance(shape, int) or np.ndim(shape) == 0
            shape = shape[0] if np.ndim(shape) > 0 else shape
            nside = (shape / 12) ** 0.5
        if int(nside) != nside:
            raise TypeError(f"invalid nside {nside!r}; expected int")
        if nest is not True:
            raise NotImplementedError("only nested order currently supported")
        assert isinstance(splits, int) or np.ndim(splits) == 0
        splits = splits[0] if np.ndim(splits) > 0 else splits
        if not (
            (splits == 1 or splits % 4 == 0)
            and (splits == 1 or splits % 4 == 0 or parent_splits is None)
        ):
            raise AssertionError()
        self.nside = int(nside)
        self.nest = nest
        size = 12 * nside**2
        if not isinstance(fill_strategy, str):
            raise TypeError(f"invalid fill_strategy {fill_strategy!r}")
        if fill_strategy.lower() not in ("same", "unique"):
            raise ValueError(f"invalid fill_strategy value {fill_strategy!r}")
        self.fill_strategy = fill_strategy.lower()
        super().__init__(shape=size, splits=splits, parent_splits=parent_splits)

    def neighborhood(self, index, window_size: int, ensemble_axis=None):
        from healpy.pixelfunc import get_all_neighbours

        if ensemble_axis is not None:
            raise NotImplementedError()

        dtp = np.result_type(index)
        if window_size not in (1, 9, self.size):
            nie = "only zero, 1st and all neighbors allowed for now"
            raise NotImplementedError(nie)
        if window_size == self.size:
            assert ~np.any(index < 0)
            return np.add.outer(index, np.arange(self.size, dtype=dtp)) % self.size

        index_shape = np.shape(index)
        index = np.ravel(index)
        n_pix = np.size(index)
        neighbors = np.zeros((n_pix, window_size), dtype=int)  # can contain `-1`
        neighbors[:, 0] = index
        nbr = get_all_neighbours(self.nside, index, nest=self.nest)
        nbr = nbr.reshape(window_size - 1, n_pix).T
        neighbors[:, 1:] = nbr

        if self.fill_strategy == "unique":
            neighbors = _fill_bad_healpix_neighbors(neighbors)
        elif self.fill_strategy == "same":
            (bad_indices,) = np.nonzero(np.any(neighbors == -1, axis=1))
            for i in bad_indices:
                neighbors[i][neighbors[i] == -1] = neighbors[i, 0]
        else:
            raise AssertionError()

        neighbors = np.squeeze(neighbors, axis=0) if index_shape == () else neighbors
        neighbors = neighbors.reshape(index_shape + (window_size,))
        return neighbors.astype(dtp)[np.newaxis]

    def index2coord(self, index, **kwargs):
        from healpy.pixelfunc import pix2vec

        shp = index.shape[1:]
        cc = pix2vec(self.nside, np.ravel(index), nest=self.nest)
        return np.stack(cc, axis=0).reshape((3,) + shp)

    def coord2index(self, coord, **kwargs):
        return NotImplementedError()

    def index2volume(self, index, **kwargs):
        r = 1.0
        surface = 4 * np.pi * r**2
        return (surface / self.size)[(np.newaxis,) * index.ndim]


class HEALPixGrid(Grid):
    def __init__(
        self,
        *,
        nside0: Optional[int] = None,
        depth: Optional[int] = None,
        nest=True,
        shape0=None,
        splits=None,
        fill_strategy: str = "same",
    ):
        self.nest = nest
        if shape0 is not None:
            assert nside0 is None
            assert isinstance(shape0, int) or np.ndim(shape0) == 0
            shape0 = shape0[0] if np.ndim(shape0) > 0 else shape0
            nside0 = (shape0 / 12) ** 0.5
            assert int(nside0) == nside0
            nside0 = int(nside0)
        self.nside0 = nside0
        if splits is None:
            splits = (4,) * depth
        super().__init__(
            shape0=12 * self.nside0**2,
            splits=splits,
            atLevel=partial(HEALPixGridAtLevel, fill_strategy=fill_strategy),
        )

    def amend(self, splits=None, *, added_depth: Optional[int] = None):
        if added_depth is not None and splits is not None:
            ve = "only one of `additional_depth` and `splits` allowed"
            raise ValueError(ve)
        if added_depth is not None:
            splits = (4,) * added_depth
        else:
            assert splits is not None
            splits = (splits,) if isinstance(splits, int) else splits
        splits = tuple(np.atleast_1d(s) for s in splits)
        return self.__class__(
            nside0=self.nside0, shape0=self.shape0, splits=self.splits + splits
        )


@dataclass()
class OGridAtLevel(GridAtLevel):
    grids: tuple[GridAtLevel]

    def __init__(self, *grids):
        self.grids = tuple(grids)

    @property
    def shape(self):
        return reduce(operator.add, (g.shape for g in self.grids))

    @property
    def ngrids(self):
        return len(self.grids)

    def children(self, index) -> np.ndarray:
        ndims_off = tuple(np.cumsum(tuple(g.ndim for g in self.grids)))
        ndims_sum = ndims_off[-1]
        islice = tuple(slice(l, r) for l, r in zip((0,) + ndims_off[:-1], ndims_off))
        children = tuple(g.children(index[i]) for i, g in zip(islice, self.grids))
        # Make initial entry broadcast-able to the full final shape
        out = children[0].reshape(
            (slice(None),) * children[0].ndim
            + (np.newaxis,) * (ndims_sum - self.grids[0].ndim)
        )
        # Successively concatenate all broadcasted children
        for c, i in zip(children[1:], islice[1:]):
            c = c.reshape(
                (slice(None),) * index.ndim
                + (np.newaxis,) * i.start
                + (slice(None),) * (i.stop - i.start)
                + (np.newaxis,) * (ndims_sum - i.stop)
            )
            assert c.shape[0] == (i.stop - i.start)
            bshp = np.broadcast_shapes(out.shape[1:], c.shape[1:])
            c = np.broadcast_to(c, c.shape[:1] + bshp)
            out = np.broadcast_to(out, out.shape[:1] + bshp)
            np.concatenate((out, c), axis=0)
        return out

    def neighborhood(self, index, window_size: tuple[int]):
        ndims_off = tuple(np.cumsum(tuple(g.ndim for g in self.grids)))
        ndims_sum = ndims_off[-1]
        islice = tuple(slice(l, r) for l, r in zip((0,) + ndims_off[:-1], ndims_off))
        window_size = (
            (window_size,) * self.ngrids
            if isinstance(window_size, int)
            else window_size
        )
        assert len(window_size) == self.ngrids
        neighborhood = tuple(
            g.neighborhood(index[i], wsz)
            for i, g, wsz in zip(islice, self.grids, window_size)
        )
        # Make initial entry broadcast-able to the full final shape
        out = neighborhood[0].reshape(
            (slice(None),) * neighborhood[0].ndim
            + (np.newaxis,) * (ndims_sum - self.grids[0].ndim)
        )
        # Successively concatenate all broadcasted neighbors
        for n, i in zip(neighborhood[1:], islice[1:]):
            n = n.reshape(
                (slice(None),) * index.ndim
                + (np.newaxis,) * i.start
                + (slice(None),) * (i.stop - i.start)
                + (np.newaxis,) * (ndims_sum - i.stop)
            )
            assert n.shape[0] == (i.stop - i.start)
            bshp = np.broadcast_shapes(out.shape[1:], n.shape[1:])
            n = np.broadcast_to(n, n.shape[:1] + bshp)
            out = np.broadcast_to(out, out.shape[:1] + bshp)
            np.concatenate((out, n), axis=0)
        return out

    def parent(self, index):
        ndims_off = tuple(np.cumsum(tuple(g.ndim for g in self.grids)))
        islice = tuple(slice(l, r) for l, r in zip((0,) + ndims_off[:-1], ndims_off))
        parent = tuple(g.parent(index[i]) for i, g in zip(islice, self.grids))
        return np.stack(parent, axis=0)

    def index2coord(self, index, **kwargs):
        ndims_off = tuple(np.cumsum(tuple(g.ndim for g in self.grids)))
        islice = tuple(slice(l, r) for l, r in zip((0,) + ndims_off[:-1], ndims_off))
        coord = tuple(g.index2coord(index[i]) for i, g in zip(islice, self.grids))
        return np.concatenate(coord, axis=0)

    def coord2index(self, coord, **kwargs):
        # TODO[@Philipp+@Gordian]: The grids do not store the ndim of the coord
        # array, thus we can not split the total coord easily
        return NotImplementedError()

    def index2volume(self, index, **kwargs):
        ndims_off = tuple(np.cumsum(tuple(g.ndim for g in self.grids)))
        islice = tuple(slice(l, r) for l, r in zip((0,) + ndims_off[:-1], ndims_off))
        volume = tuple(g.index2volume(index[i]) for i, g in zip(islice, self.grids))
        return reduce(operator.mul, volume)


@dataclass()
class OGrid(Grid):
    grids: tuple[Grid]

    def __init__(self, *grids):
        self.grids = tuple(grids)
        self.depth = self.grids[0].depth
        for i, g in enumerate(grids):
            if not isinstance(g, Grid):
                raise TypeError(f"Grid {g.__name__} at index {i} of invalid type")
            if g.depth != self.depth:
                raise ValueError(f"Grid {g.__name__} at index {i} not of same depth")
        self.atLevel = OGridAtLevel

    def amend(self, splits):
        splits = (splits,) if isinstance(splits, int) else splits
        splits = tuple(np.atleast_1d(s) for s in splits)
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


class FlatGridAtLevel(GridAtLevel):
    """Same as :class:`GridAtLevel` but with a single global integer index for each voxel."""

    gridAtLevel: GridAtLevel
    gridshape0: npt.NDArray[np.int_]
    grid_all_splits: tuple[npt.NDArray[np.int_]]
    ordering: str

    def __init__(self, gridAtLevel, gridshape0, grid_all_splits, ordering="nest"):
        if not isinstance(gridAtLevel, GridAtLevel):
            raise TypeError(f"Grid {gridAtLevel.__name__} of invalid type")
        self.gridAtLevel = gridAtLevel
        self.ordering = ordering
        if ordering not in ["serial", "nest"]:
            raise ValueError(f"Unknown flat index ordering scheme {ordering}")

        self.gridshape0 = np.asarray(gridshape0)
        self.grid_all_splits = tuple(np.atleast_1d(s) for s in grid_all_splits)
        super().__init__(
            shape=(reduce(operator.mul, gridAtLevel.shape, 1),),
            splits=None,
            parent_splits=None,
        )

    def _weights_serial(self, levelshift):
        shape = (self.gridshape0,) + self.grid_all_splits
        if levelshift == 0:
            shape = shape[:-1]
        elif levelshift == -1:
            shape = shape[:-2]
        else:
            if levelshift != 1:
                raise ValueError(f"Inconsistent shift in level: {levelshift}")
        shape = reduce(operator.mul, shape)
        wgt = np.append(shape[1:], 1)
        return np.cumprod(wgt[::-1])[::-1]

    def _weights_nest(self, levelshift):
        wgts = (self.gridshape0,) + self.grid_all_splits
        if levelshift == 0:
            wgts = wgts[:-1]
        elif levelshift == -1:
            wgts = wgts[:-2]
        else:
            if levelshift != 1:
                raise ValueError(f"Inconsistent shift in level: {levelshift}")
        return np.stack(wgts, axis=0)

    def index_to_flatindex(self, index, levelshift=0):
        if self.ordering == "serial":
            wgt = self._weights_serial(levelshift)
            wgt = wgt[(slice(None),) + (np.newaxis,) * (index.ndim - 1)]
            return (wgt * index).sum(axis=0).astype(index)[np.newaxis, ...]
        if self.ordering == "nest":
            # TODO vectorize better
            fid = np.zeros(index.shape[1:], dtype=index.dtype)
            wgts = self._weights_nest(levelshift)
            for n, ww in enumerate(wgts):
                j = 0
                for ax in range(ww.size):
                    j *= ww[ax]
                    j += (index[ax] // wgts[(n + 1) :, ax].prod()) % ww[ax]
                fid *= ww.prod()
                fid += j
            return fid[np.newaxis, ...]
        raise RuntimeError

    def flatindex_to_index(self, index, levelshift=0):
        # TODO vectorize better
        if self.ordering == "serial":
            wgt = self._weights_serial(levelshift)
            tm = np.copy(index[0])
            index = np.zeros(wgt.shape + index.shape[1:], dtype=index.dtype)
            for i, w in enumerate(wgt):
                index[i] = tm // w
                tm -= w * index[i]
            return index.astype(index.dtype)
        if self.ordering == "nest":
            wgts = self._weights_nest(levelshift)
            fid = np.copy(index[0])
            index = np.zeros((wgts.shape[0],) + index.shape[1:], dtype=index.dtype)
            for n, ww in reversed(list(enumerate(ww))):
                fct = ww.prod()
                j = fid % fct
                for ax in range(ww.size)[::-1]:
                    index[ax] += wgts[(n + 1) :, ax].prod() * (j % ww[ax])
                    j //= ww[ax]
                fid //= fct
            return index

        raise RuntimeError

    def children(self, index) -> np.ndarray:
        index = self._parse_index(index)
        index = self.flatindex_to_index(index)
        children = self.gridAtLevel.children(index)
        return self.index_to_flatindex(children, +1)

    def neighborhood(self, index, window_size: Iterable[int], ensemble_axis=None):
        index = self._parse_index(index)
        index = self.flatindex_to_index(index)
        window = self.gridAtLevel.neighborhood(
            index, window_size=window_size, ensemble_axis=ensemble_axis
        )
        return self.index_to_flatindex(window)

    def parent(self, index):
        index = self._parse_index(index)
        index = self.flatindex_to_index(index)
        window = self.gridAtLevel.parent(index)
        return self.index_to_flatindex(window, -1)

    def index2coord(self, index, **kwargs):
        index = self._parse_index(index)
        index = self.flatindex_to_index(index)
        return self.gridAtLevel.index2coord(index, **kwargs)

    def coord2index(self, coord, **kwargs):
        index = self.gridAtLevel.coord2index(coord, **kwargs)
        return self.index_to_flatindex(index)

    def index2volume(self, index, **kwargs):
        index = self._parse_index(index)
        index = self.flatindex_to_index(index)
        return self.gridAtLevel.index2volume(index, **kwargs)


class FlatGrid(Grid):
    """Same as :class:`Grid` but with a single global integer index for each voxel."""

    grid: Grid
    ordering: str

    def __init__(self, grid, ordering="nest"):
        if not isinstance(grid, Grid):
            raise TypeError(f"Grid {grid.__name__} of invalid type")
        self.grid = grid
        if ordering not in ["serial", "nest"]:
            raise ValueError(f"Unknown flat index ordering scheme {ordering}")
        self.ordering = ordering
        super().__init__(
            shape0=grid.shape0, splits=grid.splits, atLevel=FlatGridAtLevel
        )

    def amend(self, splits):
        grid = self.grid.amend(splits)
        return self.__class__(grid=grid, ordering=self.ordering)

    def at(self, level: int):
        level = self._parse_level(level)
        gridAtLevel = self.grid.at(level)
        return self.atLevel(
            gridAtLevel,
            self.grid.shape0,
            self.grid.splits[:level],
            ordering=self.ordering,
        )


class SparseGridAtLevel(FlatGridAtLevel):
    mapping: npt.NDArray[np.int_]
    parent_mapping: Optional[npt.NDArray[np.int_]] = None
    children_mapping: Optional[npt.NDArray[np.int_]] = None

    def __init__(
        self,
        gridAtLevel,
        gridshape0,
        grid_all_splits,
        mapping,
        ordering="nest",
        parent_mapping=None,
        children_mapping=None,
    ):
        if not isinstance(gridAtLevel, GridAtLevel):
            raise TypeError(f"Grid {gridAtLevel.__name__} of invalid type")
        self.mapping = mapping
        self.parent_mapping = parent_mapping
        self.children_mapping = children_mapping
        super().__init__(
            gridAtLevel=gridAtLevel,
            gridshape0=gridshape0,
            grid_all_splits=grid_all_splits,
            ordering=ordering,
        )
        self.shape = (self.mapping.size,)

    def _mapping(self, levelshift):
        if levelshift == -1:
            mapping = self.parent_mapping
        elif levelshift == 0:
            mapping = self.mapping
        elif levelshift == 1:
            mapping = self.children_mapping
        else:
            raise ValueError(f"Inconsistent shift in level: {levelshift}")
        return mapping

    def flatindex_to_index(self, index, levelshift=0):
        index = self._mapping(levelshift)[index]
        return super().flatindex_to_index(index, levelshift)

    def index_to_flatindex(self, index, levelshift=0):
        # FIXME global vs local dicts?
        mapping = {fid: i for i, fid in enumerate(self._mapping(levelshift))}
        shp = index.shape
        index = np.array(mapping[ii] for ii in index.ravel()).reshape(shp)
        return super().index_to_flatindex(index, levelshift)

    def toFlatGridAtLevel(self):
        return FlatGridAtLevel(
            self.gridAtLevel,
            self.shape0,
            self.all_parent_splits,
            ordering=self.ordering,
        )


class SparseGrid(Grid):
    """Realized :class:`FlatGrid` keeping track of the indices that are actually
    being modeled at the end of the day. This class is especially convenient for
    open boundary conditions but works for arbitrarily sparsely resolved grids."""

    mapping: tuple[npt.NDArray[np.int_]]

    def __init__(self, grid, mapping, ordering="nest"):
        if not isinstance(grid, Grid):
            raise TypeError(f"Grid {grid.__name__} of invalid type")
        self.grid = grid
        self.ordering = ordering

        mapping = (mapping,) if not isinstance(mapping, tuple) else mapping
        mapping = tuple(np.atleast_1d(m) for m in mapping)

        if len(mapping) != grid.depth:
            md, gd = len(mapping), grid.depth
            msg = f"Map depth {md} does not match grid {grid.__name__} depth {gd}"
            raise ValueError(msg)
        super().__init__(
            shape0=grid.shape0, splits=grid.splits, atLevel=SparseGridAtLevel
        )

    def amend(self, splits, mapping, **kwargs):
        grid = self.grid.amend(splits, **kwargs)
        mapping = (mapping,) if not isinstance(mapping, tuple) else mapping
        return self.__class__(grid, mapping, nest=self.nest)

    def at(self, level: int):
        level = self._parse_level(level)
        gridAtLevel = self.grid.at(level)
        parent_mapping = None if level == 0 else self.mapping[level - 1]
        children_mapping = None if level == self.depth - 1 else self.mapping[level + 1]
        return self.atLevel(
            gridAtLevel,
            self.grid.shape0,
            self.grid.splits[: (level + 1)],
            self.mapping[level],
            ordering=self.ordering,
            parent_mapping=parent_mapping,
            children_mapping=children_mapping,
        )
