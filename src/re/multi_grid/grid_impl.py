from dataclasses import dataclass
from functools import partial
from typing import Iterable, Optional

import jax.numpy as jnp
import numpy as np
from jax import vmap

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
        from .jhealpix_reference import get_all_neighbours_valid

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
            f = partial(get_all_neighbours_valid, self.nside, nest=self.nest)
            for _ in range(index.ndim - 1):
                f = vmap(f)
            nbrs = f(index[0])[jnp.newaxis, ...]
            return jnp.concatenate((index[..., jnp.newaxis], nbrs), axis=-1).astype(dtp)
        nie = "only zero, 1st and all neighbors allowed for now"
        raise NotImplementedError(nie)

    def index2coord(self, index, **kwargs):
        from .jhealpix_reference import pix2vec

        assert index.shape[0] == 1
        f = partial(pix2vec, self.nside, nest=self.nest)
        for _ in range(index.ndim - 1):
            f = vmap(f)
        cc = f(index[0])
        return jnp.stack(cc, axis=0)

    def coord2index(self, coord, dtype=np.uint64, **kwargs):
        from .jhealpix_reference import vec2pix

        assert coord.shape[0] == 3
        f = partial(vec2pix, self.nside, nest=self.nest)
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
        depth: Optional[int] = None,
        nest=True,
        shape0=None,
        splits=None,
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


class CartesianGridAtLevel(OpenGridAtLevel):
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


class CartesianGrid(OpenGrid):
    def __init__(
        self,
        *,
        min_shape,
        window_size,
        distances=None,
        splits=2,
        depth=None,
        desired_shape0=128,
    ):
        """Create a regular Cartesian grid with a given minimum shape and a default
        volume of unity at the final depth.

        The initialization automatically determines a suitable depth and padding
        (using the `window_size`) if they are unspecified.

        Amending the grid will increase the resolution while keeping the previous
        grids including their pixel distances the same. Due to padding, the amended
        grids will live on slightly smaller volume than the previous grids and will
        not anymore start exactly at zero.
        """
        if np.ndim(splits) != 2:
            if depth is None:
                desired_shape0 = np.broadcast_to(desired_shape0, np.shape(min_shape))
                splits = np.broadcast_to(splits, np.shape(min_shape))
                depth = max(
                    np.emath.logn(splits, min_shape)
                    - np.emath.logn(splits, desired_shape0)
                )
                depth = max(int(np.ceil(depth)), 0)
            splits = np.broadcast_to(splits, (depth,) + np.shape(min_shape))
        padding = np.ceil((window_size - 1) // 2).astype(np.int_)
        padding = np.broadcast_to(padding, (depth,) + np.shape(min_shape))

        # Conservative estimate of the shape at zero depth
        shape0 = np.ceil(
            np.array(min_shape) / np.prod(splits, axis=0, initial=1)
            + (2 + 2 / np.min(splits, axis=0, initial=1))
            * np.max(padding, axis=0, initial=0)
            + 1
        ).astype(np.int_)
        # Exact final shape assuming the above conservative `shape0`
        shape, shifts = shape0, np.zeros_like(shape0, dtype=float)
        for si, pd in zip(splits, padding):
            shape = si * (shape - 2 * pd)
            shifts = si * (shifts + pd)
        self.shifts0 = shifts / np.prod(splits, axis=0, initial=1)
        distances = 1.0 / shape if distances is None else distances
        self.distances0 = np.atleast_1d(distances) * np.prod(splits, axis=0, initial=1)
        super().__init__(
            shape0=shape0,
            splits=splits,
            padding=padding,
            atLevel=partial(
                CartesianGridAtLevel,
                shifts0=self.shifts0,
                distances0=self.distances0,
                all_splits=splits,
            ),
        )

    def amend(self, splits, padding):
        splits = (splits,) if isinstance(splits, int) else splits
        splits = tuple(np.broadcast_to(s, self.shape0.shape) for s in splits)
        padding = (padding,) if isinstance(padding, int) else padding
        padding = tuple(np.broadcast_to(p, self.shape0.shape) for p in padding)
        return self.__class__(
            shape0=self.shape0,
            splits=self.splits + splits,
            padding=self.padding + padding,
            atLevel=partial(self.atLevel, all_splits=self.splits + splits),
        )


def logaritmic_grid(depth, rshape0, rlim):
    gr_r = OpenGrid(shape0=(rshape0,), splits=(2,) * depth, padding=(1,) * depth)
    grd = gr_r.at(gr_r.depth)
    Nr = grd.size
    ls = grd.index2coord(np.arange(grd.size))
    print("Log-grid size:", Nr)

    def l_to_r(l, lmi, lma, rmi, rma):
        dl = lma - lmi
        lp = (l - lmi) / dl
        b = np.log(rmi)
        a = np.log(rma) - b
        return jnp.exp(a * lp + b)

    def r_to_l(r, lmi, lma, rmi, rma):
        dl = lma - lmi
        b = np.log(rmi)
        a = np.log(rma) - b

        lp = (jnp.log(r) - b) / a
        return lp * dl + lmi

    f_rl = partial(l_to_r, lmi=ls[0], lma=ls[-1], rmi=rlim[0], rma=rlim[1])
    f_lr = partial(r_to_l, lmi=ls[0], lma=ls[-1], rmi=rlim[0], rma=rlim[1])

    assert np.allclose(ls, f_lr(f_rl(ls)))
    assert np.allclose(rlim[0], f_rl(ls[0]))
    assert np.allclose(rlim[1], f_rl(ls[-1]))

    class MyLGrid(OpenGridAtLevel):
        def coord2index(self, coord):
            coord = f_lr(coord)
            return super().coord2index(coord)

        def index2coord(self, index):
            coord = super().index2coord(index)
            return f_rl(coord)

        def index2volume(self, index):
            coords = super().index2coord(index + jnp.array([-0.5, 0.5])[:, None])
            return jnp.prod(coords[1] - coords[0], axis=0)

    return OpenGrid(
        shape0=gr_r.shape0, splits=gr_r.splits, padding=gr_r.padding, atLevel=MyLGrid
    )


def get_hplogr_grid(depth, rshape0, nside0, rlim):
    gr_r = logaritmic_grid(depth, rshape0, rlim)
    gr_hp = HEALPixGrid(nside0=nside0, depth=depth)
    print("HP Nside:", gr_hp.at(gr_hp.depth).nside)

    class MyMGrid(MGridAtLevel):
        def index2coord(self, index, **kwargs):
            coords = super().index2coord(index, **kwargs)
            return coords[:1] * coords[1:]

        def coord2index(self, coord, **kwargs):
            assert coord.shape[0] == 3
            r = jnp.linalg.norm(coord, axis=0)[jnp.newaxis, ...]
            coord = jnp.concatenate((r, coord / r), axis=0)
            return super().coord2index(coord, **kwargs)

    gr = MGrid(gr_r, gr_hp, atLevel=MyMGrid)
    print("Base size:", gr.at(0).size)
    return gr
