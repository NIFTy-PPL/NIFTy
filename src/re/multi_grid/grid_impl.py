from dataclasses import dataclass
from functools import partial
from typing import Iterable, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import vmap

# NOTE, jhealpix must be a top-level import so that the global jnp.ndarray defined
# therein are defined at import time instead during jax.jit as this would lead to
# seemingly leaky tracers.
from . import jhealpix
from .grid import Grid, GridAtLevel, MGrid, MGridAtLevel, OpenGrid, OpenGridAtLevel


@dataclass()
class HEALPixGridAtLevel(GridAtLevel):
    nside: int
    nest: bool
    fill_strategy: str

    def __init__(
        self,
        shape=None,
        splits=None,
        parent_splits=None,
        *,
        nside: int = None,
        nest=True,
    ):
        if shape is not None:
            assert nside is None
            assert np.ndim(shape) == 1 and shape.size == 1
            nside = (shape[0] / 12) ** 0.5
        if int(nside) != nside:
            raise TypeError(f"invalid nside {nside!r}; expected int")
        if nest is not True:
            raise NotImplementedError("only nested order currently supported")
        if splits is not None:
            splits = np.atleast_1d(splits)
            assert np.ndim(splits) == 1 and splits.size == 1
            if not (splits[0] == 1 or splits[0] % 4 == 0):
                raise AssertionError()
        if parent_splits is not None:
            parent_splits = np.atleast_1d(parent_splits)
            assert np.ndim(parent_splits) == 1 and parent_splits.size == 1
            if not (parent_splits[0] == 1 or parent_splits[0] % 4 == 0):
                raise AssertionError()
        self.nside = int(nside)
        self.nest = nest
        size = 12 * self.nside**2
        super().__init__(shape=size, splits=splits, parent_splits=parent_splits)

    def neighborhood(self, index, window_size: Iterable[int]):
        index = jnp.atleast_1d(index)
        if not isinstance(window_size, int):
            window_size = window_size[0]
        assert index.shape[0] == 1
        dtp = jnp.result_type(index)
        if window_size == 1:
            return index[..., jnp.newaxis]
        if window_size == self.size:
            assert np.all(index >= 0) and np.all(index < self.size)
            nbrs = np.arange(self.size, dtype=dtp)
            nbrs = nbrs[(np.newaxis,) * index.ndim + (slice(None),)]
            return (index[..., jnp.newaxis] + nbrs) % self.size
        if window_size == 9:
            f = partial(jhealpix.get_all_neighbours_valid, self.nside, nest=self.nest)
            for _ in range(index.ndim - 1):
                f = vmap(f)
            nbrs = f(index[0])[jnp.newaxis, ...]
            return jnp.concatenate((index[..., jnp.newaxis], nbrs), axis=-1).astype(dtp)
        nie = "only zero, 1st and all neighbors allowed for now"
        raise NotImplementedError(nie)

    def index2coord(self, index, **kwargs):
        assert index.shape[0] == 1
        f = partial(jhealpix.pix2vec, self.nside, nest=self.nest)
        for _ in range(index.ndim - 1):
            f = vmap(f)
        cc = f(index[0])
        return jnp.stack(cc, axis=0)

    def coord2index(self, coord, dtype=np.uint64, **kwargs):
        assert coord.shape[0] == 3
        f = partial(jhealpix.vec2pix, self.nside, nest=self.nest)
        for _ in range(coord.ndim - 1):
            f = vmap(f)
        idx = f(*(cc for cc in coord))
        return (idx[jnp.newaxis, ...]).astype(dtype)

    def index2volume(self, index, **kwargs):
        r = 1.0
        surface = 4 * np.pi * r**2
        return (surface / self.size)[(np.newaxis,) * index.ndim]


@dataclass()
class HEALPixGrid(Grid):
    def __init__(
        self,
        *,
        nside0: Optional[int] = None,
        nside: Optional[int] = None,
        depth: Optional[int] = None,
        nest=True,
        shape0=None,
        splits=None,
    ):
        """HEALPix pixelization grid.

        The grid can be defined either via the final nside, the initial nside
        (nside0), or the initial shape (shape0).
        """
        self.nest = nest
        if shape0 is not None:
            assert nside0 is None
            assert isinstance(shape0, int) or np.ndim(shape0) == 0
            shape0 = shape0[0] if np.ndim(shape0) > 0 else shape0
            nside0 = (shape0 / 12) ** 0.5
            assert int(nside0) == nside0
            nside0 = int(nside0)
        if nside is not None:
            assert nside0 is None
            assert depth is not None
            nside0 = nside / 2**depth
            assert int(nside0) == nside0
        self.nside0 = nside0
        assert self.nside0 > 0
        if splits is None:
            splits = (4,) * depth
        super().__init__(
            shape0=12 * self.nside0**2,
            splits=splits,
            atLevel=partial(HEALPixGridAtLevel, nest=self.nest),
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
        return self.__class__(nside0=self.nside0, splits=self.splits + splits)


class SimpleOpenGridAtLevel(OpenGridAtLevel):
    def __init__(
        self,
        shape,
        splits=None,
        parent_splits=None,
        *,
        shifts0,
        distances0,
        all_splits,
        level=None,
        shifts=None,
        **kwargs,
    ):
        assert level <= len(all_splits)
        shifts = shifts - shifts0 * np.prod(all_splits[:level], axis=0, initial=1.0)
        self.distances = distances0 / np.prod(all_splits[:level], axis=0, initial=1.0)
        super().__init__(
            shape, splits=splits, parent_splits=parent_splits, shifts=shifts, **kwargs
        )

    def index2coord(self, index):
        bc = (slice(None),) + (np.newaxis,) * (index.ndim - 1)
        coord = super().index2coord(index)
        return coord * ((self.shape + 2 * self.shifts) * self.distances)[bc]

    def coord2index(self, coord, dtype=np.uint64):
        bc = (slice(None),) + (np.newaxis,) * (coord.ndim - 1)
        coord = coord / ((self.shape + 2 * self.shifts) * self.distances)[bc]
        return super().coord2index(self, coord, dtype=dtype)

    def index2volume(self, index):
        vol = super().index2volume(index)
        return vol * np.prod((self.shape + 2 * self.shifts) * self.distances)


def SimpleOpenGrid(
    *,
    min_shape: Tuple[int],
    window_size: Union[int, Tuple[int]] = 3,
    splits: Union[int, Tuple[int], Tuple[Tuple[int]]] = 2,
    distances: Optional[Union[float, Tuple[float]]] = None,
    depth: Optional[int] = None,
    desired_size0: int = 128,
    atLevel: GridAtLevel = SimpleOpenGridAtLevel,
) -> OpenGrid:
    """Create a regular Cartesian grid with a given minimum shape and a default
    volume of unity at the final depth.

    The initialization automatically determines a suitable depth and padding
    (using the `window_size`).

    Amending the grid will increase the resolution while keeping the previous
    grids including their pixel distances the same. Due to padding, the amended
    grids will live on a slightly smaller volume than the previous grids and won't
    start exactly at zero.

    Parameters
    ----------
    min_shape:
        Minimum shape along each dimension. If the shape can be larger than requested
        at no additional costs, it will be automatically extended.
    window_size:
        Size of the refinement window. The larger the window, the more accurate but
        more costly the refinement.
    splits:
        Number of child pixels of a parent pixels.
    distances:
        Distance between pixels if the grid is regular.
    depth:
        Refinement depth.
    desired_size0:
        Approximate size of the initial coarse-most grid.
    """
    min_shape = np.atleast_1d(min_shape)
    if np.ndim(splits) != 2:
        if depth is None:
            desired_shape0 = desired_size0 ** (1.0 / min_shape.ndim)
            desired_shape0 = np.ceil(desired_shape0).astype(np.int_)
            desired_shape0 = np.broadcast_to(desired_shape0, min_shape.shape)
            desired_shape0 = np.min(
                (desired_shape0, (min_shape / 1.5).astype(int)), axis=0
            )
            splits = np.broadcast_to(splits, min_shape.shape)
            depth = max(
                np.emath.logn(splits, min_shape) - np.emath.logn(splits, desired_shape0)
            )
            depth = max(int(np.ceil(depth)), 0)
        splits = np.broadcast_to(splits, (depth,) + min_shape.shape)
    padding = np.ceil((window_size - 1) // 2).astype(np.int_)
    padding = np.broadcast_to(padding, (depth,) + min_shape.shape)

    # Conservative estimate of the shape at zero depth
    shape0 = np.ceil(
        min_shape / np.prod(splits, axis=0, initial=1)
        + (2 + 2 / np.min(splits, axis=0, initial=1))
        * np.max(padding, axis=0, initial=0)
        + 1
    ).astype(np.int_)
    # Exact final shape assuming the above conservative `shape0`
    shape, shifts = shape0, np.zeros_like(shape0, dtype=float)
    for si, pd in zip(splits, padding):
        shape = si * (shape - 2 * pd)
        shifts = si * (shifts + pd)
    shifts0 = shifts / np.prod(splits, axis=0, initial=1)
    distances = 1.0 / shape if distances is None else distances
    distances0 = np.atleast_1d(distances) * np.prod(splits, axis=0, initial=1)
    return OpenGrid(
        shape0=shape0,
        splits=splits,
        padding=padding,
        atLevel=partial(atLevel, shifts0=shifts0, distances0=distances0),
    )


class LogarithmicGridAtLevel(SimpleOpenGridAtLevel):
    def __init__(self, *args, coord_offset, coord_scale, **kwargs):
        # NOTE, technically `coord_offset` and `coord_scale` are redundant with
        # `shifts` and `distances`, however, for ease of use, we first let them
        # scale the grid to (0, 1).
        self.coord_offset = coord_offset
        self.coord_scale = coord_scale
        super().__init__(*args, **kwargs)

    @property
    def r_min(self):
        return self.index2coord(np.array([-0.5]))

    @property
    def r_max(self):
        return self.index2coord(np.array([self.shape[0] - 0.5]))

    def index2coord(self, index):
        coord = super().index2coord(index)
        return jnp.exp(self.coord_scale * coord + self.coord_offset)

    def coord2index(self, coord, dtype=np.uint64):
        coord = (jnp.log(coord) - self.coord_offset) / self.coord_scale
        return super().coord2index(self, coord, dtype=dtype)

    def index2volume(self, index):
        a = (slice(None),) + (np.newaxis,) * index.ndim
        coords = super().index2coord(index + jnp.array([-0.5, 0.5])[a])
        return jnp.prod(coords[1] - coords[0], axis=0)


def LogarithmicGrid(
    *,
    r_min: float,
    r_max: float,
    distances=None,
    **kwargs,
) -> OpenGrid:
    """Logarithmic grid on top of `SimpleOpenGrid` spanning from `r_min` to `r_max`
    at the final depth.
    """
    if distances is not None:
        raise ValueError("`distances` are incompatible with a logarithmic grid")
    if r_min < 0.0 or r_max < r_min:
        raise ValueError(f"invalid r_min {r_min!r} or r_max {r_max!r}")
    coord_offset = np.log(r_min)
    coord_scale = np.log(r_max) - coord_offset
    return SimpleOpenGrid(
        **kwargs,
        atLevel=partial(
            LogarithmicGridAtLevel, coord_offset=coord_offset, coord_scale=coord_scale
        ),
    )


class HPLogRGridAtLevel(MGridAtLevel):
    def index2coord(self, index, **kwargs):
        coords = super().index2coord(index, **kwargs)
        return coords[:3] * coords[3]

    def coord2index(self, coord, **kwargs):
        assert coord.shape[0] == 3
        r = jnp.linalg.norm(coord, axis=0)[jnp.newaxis, ...]
        coord = jnp.concatenate((coord / r, r), axis=0)
        return super().coord2index(coord, **kwargs)


def HPLogRGrid(
    min_shape: Optional[Tuple[int, int]] = None,
    *,
    nside: Optional[int] = None,
    r_min_shape: Optional[int] = None,
    r_min,
    r_max,
    r_window_size=3,
    nside0=16,
    atLevel=HPLogRGridAtLevel,
) -> MGrid:
    """Meshgrid of a HEALPix grid and a logarithmic grid.

    See `HEALPixGrid` and `LogarithmicGrid`."""
    if r_min_shape is None and nside is None:
        hp_size, r_min_shape = min_shape
        nside = (hp_size / 12) ** 0.5
    depth = np.log2(nside / nside0)
    assert depth == int(depth)
    depth = int(depth)
    grid_hp = HEALPixGrid(nside0=nside0, depth=depth)
    grid_r = LogarithmicGrid(
        min_shape=r_min_shape,
        r_min=r_min,
        r_max=r_max,
        window_size=r_window_size,
        depth=depth,
    )
    return MGrid(grid_hp, grid_r, atLevel=atLevel)
