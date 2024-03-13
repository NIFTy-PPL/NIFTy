#!/usr/bin/env python3

import operator
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np

_kind_of_int = Union[tuple[int], list[int], int]


@dataclass(kw_only=True)
class GridAtLevel:
    shape: tuple[int]
    refinement_shape: tuple[int]
    parent_refinement_shape: Optional[tuple[int]] = None

    def __init__(self, shape, refinement_shape, parent_refinement_shape):
        self.shape = np.asarray(shape)
        self.refinement_shape = np.asarray(refinement_shape)
        self.parent_refinement_shape = np.asarray(parent_refinement_shape)

    def _parse_index(self, index):
        index = np.asarray(index)
        if np.any(np.any(np.abs(idx) >= s) for idx, s in zip(index, self.shape)):
            nm = self.__class__.__name__
            ve = f"index {index} is out of bounds for {nm} with shape {self.shape}"
            raise IndexError(ve)
        return index % self.shape[(slice(None),) + (np.newaxis,) * (index.ndim - 1)]

    @property
    def size(self):
        return len(self.shape)

    def children(self, index) -> np.ndarray:
        index = self._parse_index(index)
        dtp = np.result_type(index)
        rs = self.refinement_shape[(slice(None),) + (None,) * (index.ndim - 1)]
        rs_bc = (slice(None),) * index.ndim, (None,) * len(self.refinement_shape)
        c = np.mgrid[tuple(slice(0, sz) for sz in self.refinement_shape)].astype(dtp)
        c_bc = (
            (slice(None),)
            + (None,) * (index.ndim - 1)
            + (slice(None),) * len(self.refinement_shape)
        )
        return rs[rs_bc] + c[c_bc]

    def neighborhood(self, index, window_size: int, ensemble_axis=None):
        index = self._parse_index(index)
        dtp = np.result_type(index)
        # TODO
        return (
            np.add.outer(index, np.arange(window_size, dtype=dtp) - window_size // 2)
            % self.size
        )

    def parent(self, index):
        if self.parent_refinement_shape is None:
            raise IndexError("you are alone in this world")
        index = self._parse_index(index)
        bc = (slice(None),) + (np.newaxis,) * (index.ndim - 1)
        return index // self.parent_refinement_shape[bc]

    def index2coord(self, index, **kwargs):
        return NotImplementedError()

    def coord2index(self, coord, **kwargs):
        return NotImplementedError()

    def index2volume(self, index, **kwargs):
        return NotImplementedError()


class Grid:
    """Dense (single) axis with periodic boundary conditions.

    Open boundary conditions can be emulated by leaving out indices."""

    atLevel: Callable

    def __init__(
        self,
        *,
        shape0: int,
        refinement_shape: tuple[_kind_of_int],
        atLevel=GridAtLevel,
    ):
        # TODO
        self.size0 = size0
        refinement_size = (
            (refinement_size,) if isinstance(refinement_size, int) else refinement_size
        )
        self.refinement_size = tuple(refinement_size)
        self.depth = len(self.refinement_size)

    def _parse_level(self, level):
        # TODO
        if np.abs(level) >= self.depth:
            raise IndexError(f"{self.__class__.__name__} does not have level {level}")
        return level % self.depth

    def amend(self, refinement_size: _kind_of_int):
        # TODO
        refinement_size = (
            (refinement_size,) if isinstance(refinement_size, int) else refinement_size
        )
        return self.__class__(
            size0=self.size0,
            refinement_size=self.refinement_size + tuple(refinement_size),
        )

    def at(self, level: int) -> GridAtLevel:
        # TODO
        level = self._parse_level(level)
        size = self.size0 * reduce(operator.mul, self.refinement_size[:level], 1)
        rs = self.refinement_size[level]
        rs_p = self.refinement_size[level - 1] if level >= 1 else None
        return self.atLevel(
            size=size, refinement_shape=rs, parent_refinement_shape=rs_p
        )


class RegularGridAxisAtLevel(GridAtLevel):
    def index2coord():
        pass

    def index2volume():
        pass


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

    def __init__(
        self,
        *,
        nside: int,
        nest=True,
        refinement_size: int = 4,
        parent_refinement_size: Optional[_kind_of_int] = None,
    ):
        if int(nside) != nside:
            raise TypeError(f"invalid nside {nside!r}; expected int")
        if nest is not True:
            raise NotImplementedError("only nested order currently supported")
        if not (
            (refinement_size == 1 or refinement_size % 4 == 0)
            and (
                refinement_size == 1
                or refinement_size % 4 == 0
                or parent_refinement_size is None
            )
        ):
            raise AssertionError()
        self.nside = int(nside)
        self.nest = nest
        size = 12 * nside**2
        super().__init__(
            size=size,
            refinement_size=refinement_size,
            parent_refinement_size=parent_refinement_size,
        )

    def neighborhood(self, index, window_size: int, fill_strategy="same"):
        from healpy.pixelfunc import get_all_neighbours

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

        if not isinstance(fill_strategy, str):
            raise TypeError(f"invalid fill_strategy {fill_strategy!r}")
        if fill_strategy.lower() == "unique":
            neighbors = _fill_bad_healpix_neighbors(neighbors)
        elif fill_strategy.lower() == "same":
            (bad_indices,) = np.nonzero(np.any(neighbors == -1, axis=1))
            for i in bad_indices:
                neighbors[i][neighbors[i] == -1] = neighbors[i, 0]
        else:
            raise ValueError(f"invalid fill_strategy value {fill_strategy!r}")

        neighbors = np.squeeze(neighbors, axis=0) if index_shape == () else neighbors
        neighbors = neighbors.reshape(index_shape + (window_size,))
        return neighbors.astype(dtp)


class HEALPixGrid(Grid):
    def __init__(
        self,
        *,
        nside0: Optional[int] = None,
        depth: Optional[int] = None,
        nest=True,
        size0: Optional[int] = None,
        refinement_size: Optional[_kind_of_int] = None,
    ):
        self.nest = nest
        if size0 is not None:
            assert nside0 is None
            nside0 = (size0 / 12) ** 0.5
            assert int(nside0) == nside0
            nside0 = int(nside0)
        self.nside0 = nside0
        if refinement_size is None:
            refinement_size = (4,) * depth
        super().__init__(size0=12 * self.nside0**2, refinement_size=refinement_size)

    def amend(self, *, added_depth: Optional[int] = None, refinement_size=None):
        if added_depth is not None and refinement_size is not None:
            ve = "only one of `additional_depth` and `refinement_size` allowed"
            raise ValueError(ve)
        if added_depth is not None:
            rss = (4,) * added_depth
        else:
            assert refinement_size is not None
            rss = refinement_size
            rss = (rss,) if isinstance(rss, int) else rss
        return HEALPixGrid(
            nside0=self.nside0,
            size0=self.size0,
            refinement_size=self.refinement_size + rss,
        )

    def at(self, level: int) -> HEALPixGridAtLevel:
        level = self._parse_level(level)
        assert all(rs % 2 == 0 or rs == 1 for rs in self.refinement_size)
        nside = self.nside0 * reduce(
            operator.mul,
            ((rs // 2 if rs != 1 else 1) for rs in self.refinement_size[:level]),
            1,
        )
        rs = self.refinement_size[level]
        rs_p = self.refinement_size[level - 1] if level >= 1 else None
        return HEALPixGridAtLevel(
            nside=nside, refinement_size=rs, parent_refinement_size=rs_p
        )


def _stack_outer(*arrays, outer_axis=-1, stack_axis=0):
    arrays = tuple(arrays)
    ndim = len(arrays)
    outer_axis %= len(arrays[0].shape)
    window_shape = tuple(a.shape[outer_axis] for a in arrays)
    out_shp = (
        arrays[0].shape[:outer_axis]
        + window_shape
        + arrays[0].shape[(outer_axis + 1) :]
    )
    stack_axis %= len(out_shp)
    out_shp = out_shp[:stack_axis] + (ndim,) + out_shp[stack_axis:]

    res = np.zeros(out_shp, dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        a = a.reshape(
            a.shape[:outer_axis]
            + (1,) * i
            + (window_shape,)
            + (1,) * (ndim - i - 1)
            + a.shape[(outer_axis + 1) :]
        )
        res[(slice(None),) * stack_axis + (i,)] += a
    return res


def regularAxis_index2cart(index, size):
    # Map onto bincenter
    cc = (index + 0.5) / size
    # Add dummy axis for 1D space
    return cc[np.newaxis]


def healpixAxis_index2cart(index, nside, nest):
    from healpy.pixelfunc import pix2vec

    shp = index.shape[1:]
    cc = pix2vec(nside, np.ravel(index), nest=nest)
    return np.stack(cc, axis=0).reshape((3,) + shp)


def default_index2cart(index, grids):
    coords = tuple(
        (
            healpixAxis_index2cart(index[i], g.nside, g.nest)
            if isinstance(g, HEALPixGridAtLevel)
            else regularAxis_index2cart(index[i], g.size)
        )
        for i, g in enumerate(grids)
    )
    return coords  # TODO better concatenate instead of tuple?


@dataclass(kw_only=True)
class OGridAtLevel(GridAtLevel):
    grids: tuple[GridAtLevel]

    def __init__(self, *grids):
        self.grids = tuple(grids)

    @property
    def shape(self):
        return tuple(s for s in self.grids.size)

    @property
    def size(self):
        return reduce(operator.mul, self.shape, 1)

    def children(self, index) -> np.ndarray:
        window = (g.children(index[i]) for i, g in enumerate(self.grids))
        window = _stack_outer(window, outer_axis=-1, stack_axis=0)
        assert window.dtype == np.result_type(index)
        return window

    def neighborhood(self, index, window_size: tuple[int]):
        window = (
            g.neighborhood(index[i], w)
            for i, (g, w) in enumerate(zip(self.grids, window_size))
        )
        window = _stack_outer(window, outer_axis=-1, stack_axis=0)
        assert window.dtype == np.result_type(index)
        return window

    def parent(self, index):
        parid = tuple(g.parent(index[i]) for i, g in enumerate(self.grids))
        parid = np.stack(parid, axis=0)
        assert parid.dtype == np.result_type(index)
        return parid


@dataclass(kw_only=True)
class OGrid(Grid):
    grids: tuple[Grid]
    depth: int

    def __init__(self, *grids):
        self.grids = tuple(grids)
        for g in grids:
            if not isinstance(g, Grid):
                raise ValueError(f"Grid {g.__name__} not of type `GridAxis`")
        self.depth = self.grids[0].depth
        for i, g in enumerate(grids):
            if g.depth != self.depth:
                msg = f"Grid {g.__name__} at index {i} of incompatible depth {g.depth}"
                raise ValueError(msg)

    def amend(self, *grid_kwargs: tuple[dict]):
        grids = (g.amend(*kwargs) for g, kwargs in zip(self.grids, grid_kwargs))
        return OGrid(*grids, index2cart=self.index2cart)

    def at(self, level: int) -> HEALPixGridAtLevel:
        grids = (g.at(level) for g in self.grids)
        return OGridAtLevel(grids)


class FlatGridAtLevel:
    """Same as :class:`Grid` but with a single global integer index for each voxel."""

    ogrid: OGrid


class SparseGridAtLevel(FlatGridAtLevel):
    real2flat: np.ndarray
    flat2real: dict


class SparseGrid:
    """Realized :class:`FlatGridAtLevel` keeping track of the indices that are actually
    being modeled at the end of the day. This class is especially convenient for
    open boundary conditions but works for arbitrarily sparsely resolved grids."""

    real2flat: tuple[np.ndarray]
    flat2real: dict
