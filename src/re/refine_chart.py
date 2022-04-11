#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from collections import namedtuple
from functools import partial
from typing import Callable, Iterable, Literal, Optional, Tuple, Union

from jax import numpy as jnp
from jax import vmap

from .refine import _get_cov_from_loc
from .refine_util import coarse2fine_shape, fine2coarse_shape


class CoordinateChart():
    def __init__(
        self,
        min_shape: Optional[Iterable[int]] = None,
        depth: int = 7,
        *,
        shape0=None,
        _coarse_size=3,
        _fine_size=2,
        _fine_strategy: Literal["jump", "extend"] = "jump",
        rg2cart: Optional[Callable[[
            Iterable,
        ], Iterable]] = None,
        cart2rg: Optional[Callable[[
            Iterable,
        ], Iterable]] = None,
        regular_axes: Optional[Union[Iterable[int], Tuple]] = None,
        irregular_axes: Optional[Union[Iterable[int], Tuple]] = None,
        distances: Optional[Union[Iterable[float], float]] = None,
        distances0: Optional[Union[Iterable[float], float]] = None,
    ):
        """Initialize a refinement chart."""
        self.depth = depth
        if shape0 is None and min_shape is not None:
            shape0 = fine2coarse_shape(
                min_shape,
                depth,
                ceil_sizes=True,
                _coarse_size=_coarse_size,
                _fine_size=_fine_size,
                _fine_strategy=_fine_strategy
            )
        elif shape0 is None:
            raise ValueError("either `shape0` or `min_shape` must be specified")
        self.shape0 = tuple(shape0)
        self.shape = coarse2fine_shape(
            shape0,
            depth,
            _coarse_size=_coarse_size,
            _fine_size=_fine_size,
            _fine_strategy=_fine_strategy
        )

        if _fine_strategy not in ("jump", "extend"):
            ve = f"invalid `_fine_strategy`; got {_fine_strategy}"
            raise ValueError(ve)

        self.shape_at = partial(
            coarse2fine_shape,
            self.shape0,
            _coarse_size=_coarse_size,
            _fine_size=_fine_size,
            _fine_strategy=_fine_strategy
        )

        self.coarse_size = int(_coarse_size)
        self.fine_size = int(_fine_size)
        self.fine_strategy = str(_fine_strategy)

        # Derived attributes
        self.ndim = len(self.shape)

        if rg2cart is None and cart2rg is None:
            if _fine_strategy == "jump":
                fpx_in_cpx = _fine_size**depth
            elif _fine_strategy == "extend":
                fpx_in_cpx = 2**depth
            else:
                raise AssertionError()
            if distances0 is None and distances is None:
                distances = jnp.ones((self.ndim, ))
                distances0 = distances * fpx_in_cpx
            elif distances0 is not None:
                distances0 = jnp.broadcast_to(
                    jnp.atleast_1d(distances0), (self.ndim, )
                )
                distances = distances0 / fpx_in_cpx
            else:
                distances = jnp.broadcast_to(
                    jnp.atleast_1d(distances), (self.ndim, )
                )
                distances0 = distances * fpx_in_cpx

            def _rg2cart(x):
                x = jnp.asarray(x)
                return x * distances0.reshape((-1, ) + (1, ) * (x.ndim - 1))

            def _cart2rg(x):
                x = jnp.asarray(x)
                return x / distances0.reshape((-1, ) + (1, ) * (x.ndim - 1))

            if regular_axes is None and irregular_axes is None:
                regular_axes = tuple(range(self.ndim))
            self.rg2cart = _rg2cart
            self.cart2rg = _cart2rg
        elif rg2cart is not None and cart2rg is not None:
            c0 = [jnp.arange(shp) for shp in self.shape0]
            if not all(
                jnp.allclose(r, c) for r, c in zip(cart2rg(rg2cart(c0)), c0)
            ):
                raise ValueError("`cart2rg` is not the inverse of `rg2cart`")
            self.rg2cart = rg2cart
            self.cart2rg = cart2rg
            distances = distances0 = None
        else:
            ve = "invalid combination of `cart2rg`, `rg2cart` and `distances`"
            raise ValueError(ve)
        self.distances = distances
        self.distances0 = distances0

        if regular_axes is None and irregular_axes is not None:
            regular_axes = tuple(set(range(self.ndim)) - set(irregular_axes))
        elif regular_axes is not None and irregular_axes is None:
            irregular_axes = tuple(set(range(self.ndim)) - set(regular_axes))
        elif regular_axes is None and irregular_axes is None:
            regular_axes = ()
            irregular_axes = tuple(range(self.ndim))
        else:
            if set(regular_axes) | set(irregular_axes) != set(range(self.ndim)):
                ve = "`regular_axes` and `irregular_axes` do not span the full axes"
                raise ValueError(ve)
            if set(regular_axes) & set(irregular_axes) != set():
                ve = "`regular_axes` and `irregular_axes` must be exclusive"
                raise ValueError(ve)
        self.regular_axes = tuple(regular_axes)
        self.irregular_axes = tuple(irregular_axes)

        self._descr = {
            "depth": self.depth,
            "shape0": self.shape0,
            "_coarse_size": self.coarse_size,
            "_fine_size": self.fine_size,
            "_fine_strategy": self.fine_strategy,
        }
        if distances0 is not None:
            self._descr["distances0"] = tuple(distances0)
        else:
            self._descr["rg2cart"] = repr(rg2cart)
            self._descr["cart2rg"] = repr(cart2rg)

    def rgoffset(self, lvl: int) -> Tuple[float]:
        """Calculate the offset on the regular grid due to shrinking of the
        grid with increasing refinement level.

        Note
        ----
        Indices are assumed to denote the center of the pixels, i.e. the pixel
        with index `0` is assumed to be at `(0., ) * ndim`.
        """
        csz = self.coarse_size  # abbreviations for readability
        fsz = self.fine_size

        leftmost_center = 0.
        # Assume the indices denote the center of the pixels, i.e. the pixel
        # with index 0 is at (0., ) * ndim
        if self.fine_strategy == "jump":
            # for i in range(lvl):
            #     leftmost_center += ((csz - 1) / 2 - 0.5 + 0.5 / fsz) / fsz**i
            lm0 = (csz - 1) / 2 - 0.5 + 0.5 / fsz
            geo = (1. - fsz**
                   -lvl) / (1. - 1. / fsz)  # sum(fsz**-i for i in range(lvl))
            leftmost_center = lm0 * geo
        elif self.fine_strategy == "extend":
            # for i in range(lvl):
            #     leftmost_center += ((csz - 1) / 2 - 0.25 * (fsz - 1)) / 2**i
            lm0 = ((csz - 1) / 2 - 0.25 * (fsz - 1))
            geo = (1. - 2.**-lvl) * 2.  # sum(fsz**-i for i in range(lvl))
            leftmost_center = lm0 * geo
        else:
            raise AssertionError()
        return tuple((leftmost_center, ) * self.ndim)

    def ind2rg(self, indices: Iterable[Union[float, int]],
               lvl: int) -> Tuple[float]:
        """Converts pixel indices to a continuous regular grid."""
        offset = self.rgoffset(lvl)

        if self.fine_strategy == "jump":
            dvol = 1 / self.fine_size**lvl
        elif self.fine_strategy == "extend":
            dvol = 1 / 2**lvl
        else:
            raise AssertionError()
        return tuple(off + idx * dvol for off, idx in zip(offset, indices))

    def rg2ind(
        self,
        positions: Iterable[Union[float, int]],
        lvl: int,
        discretize: bool = True
    ) -> Union[Tuple[float], Tuple[int]]:
        """Converts continuous regular grid positions to pixel indices."""
        offset = self.rgoffset(lvl)

        if self.fine_strategy == "jump":
            dvol = 1 / self.fine_size**lvl
        elif self.fine_strategy == "extend":
            dvol = 1 / 2**lvl
        else:
            raise AssertionError()
        indices = tuple(pos / dvol - off for off, pos in zip(offset, positions))
        if discretize:
            indices = tuple(jnp.rint(idx).astype(jnp.int32) for idx in indices)
        return indices

    def ind2cart(self, indices: Iterable[Union[float, int]], lvl: int):
        """Computes the Cartesian coordinates of a pixel given the indices of
        it.
        """
        return self.rg2cart(self.ind2rg(indices, lvl))

    def cart2ind(self, positions, lvl, discretize=True):
        """Computes the indices of a pixel given the Cartesian coordinates of
        it.
        """
        return self.rg2ind(self.cart2rg(positions), lvl, discretize=discretize)

    def level_of(self, shape: Tuple[int]):
        """Finds the refinement level at which the number of grid points
        equate.
        """
        if not isinstance(shape, tuple):
            raise TypeError(f"invalid type of `shape`; got {type(shape)}")

        for lvl in range(self.depth + 1):
            if shape == self.shape_at(lvl):
                return lvl
        else:
            raise ValueError(f"invalid shape {shape!r}")

    def refinement_matrices(self, kernel: Callable, depth=None):
        depth = self.depth if depth is None else depth
        return coordinate_refinement_matrices(self, depth, kernel=kernel)

    def __repr__(self):
        return f"{self.__class__.__name__}(**{self._descr})"

    def __eq__(self, other):
        return repr(self) == repr(other)


def coordinate_pixel_refinement_matrices(
    chart: CoordinateChart,
    level: int,
    pixel_index: Optional[Iterable[int]] = None,
    kernel: Optional[Callable] = None,
    *,
    _cov_from_loc: Optional[Callable] = None,
):
    cov_from_loc = _get_cov_from_loc(kernel, _cov_from_loc)
    csz = int(chart.coarse_size)  # coarse size
    if csz % 2 != 1:
        raise ValueError("only odd numbers allowed for `_coarse_size`")
    fsz = int(chart.fine_size)  # fine size
    if fsz % 2 != 0:
        raise ValueError("only even numbers allowed for `_fine_size`")
    ndim = chart.ndim
    if pixel_index is None:
        pixel_index = (0, ) * ndim
    pixel_index = jnp.asarray(pixel_index)

    csz_half = int((csz - 1) / 2)
    gc = jnp.arange(-csz_half, csz_half + 1, dtype=float)
    gc = jnp.ones((ndim, 1)) * gc
    gc = jnp.stack(jnp.meshgrid(*gc, indexing="ij"), axis=-1)
    if chart.fine_strategy == "jump":
        gf = jnp.arange(fsz, dtype=float) / fsz - 0.5 + 0.5 / fsz
    elif chart.fine_strategy == "extend":
        gf = jnp.arange(fsz, dtype=float) / 2 - 0.25 * (fsz - 1)
    else:
        raise ValueError(f"invalid `_fine_strategy`; got {chart.fine_strategy}")
    gf = jnp.ones((ndim, 1)) * gf
    gf = jnp.stack(jnp.meshgrid(*gf, indexing="ij"), axis=-1)
    # On the GPU a single `cov_from_loc` call is about twice as fast as three
    # separate calls for coarse-coarse, fine-fine and coarse-fine.
    coord = jnp.concatenate(
        (gc.reshape(-1, ndim), gf.reshape(-1, ndim)), axis=0
    )
    coord = chart.ind2cart((coord + pixel_index.reshape((1, ndim))).T, level)
    coord = jnp.stack(coord, axis=-1)
    cov = cov_from_loc(coord, coord)
    cov_ff = cov[-fsz**ndim:, -fsz**ndim:]
    cov_fc = cov[-fsz**ndim:, :-fsz**ndim]
    cov_cc = cov[:-fsz**ndim, :-fsz**ndim]
    cov_cc_inv = jnp.linalg.inv(cov_cc)

    olf = cov_fc @ cov_cc_inv
    # Also see Schur-Complement
    fine_kernel = cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
    # Implicitly assume a white power spectrum beyond the numerics limit. Use
    # the diagonal as estimate for the magnitude of the variance.
    fine_kernel_fallback = jnp.diag(jnp.abs(jnp.diag(fine_kernel)))
    # Never produce NaNs (https://github.com/google/jax/issues/1052)
    fine_kernel = jnp.where(
        jnp.all(jnp.diag(fine_kernel) > 0.), fine_kernel, fine_kernel_fallback
    )
    fine_kernel_sqrt = jnp.linalg.cholesky(fine_kernel)

    return olf, fine_kernel_sqrt


RefinementMatrices = namedtuple(
    "RefinementMatrices", ("filter", "propagator_sqrt", "cov_sqrt0")
)


def coordinate_refinement_matrices(
    chart: CoordinateChart,
    depth: int,
    kernel: Callable,
    *,
    _cov_from_loc=None
):
    cov_from_loc = _get_cov_from_loc(kernel, _cov_from_loc)

    rg0_ax = [jnp.arange(shp) for shp in chart.shape0]
    rg0 = jnp.stack(jnp.meshgrid(*rg0_ax, indexing="ij"), axis=0)
    c0 = jnp.stack(chart.ind2cart(rg0, 0), axis=-1).reshape(-1, chart.ndim)
    cov_sqrt0 = jnp.linalg.cholesky(cov_from_loc(c0, c0))

    opt_lin_filter, kernel_sqrt = [], []
    olf_at = vmap(
        partial(
            coordinate_pixel_refinement_matrices,
            chart,
            kernel=None,
            _cov_from_loc=cov_from_loc,
        ),
        in_axes=(None, 0),
        out_axes=(0, 0)
    )

    for lvl in range(depth):
        shape_lvl = chart.shape_at(lvl)
        pixel_indices = []
        for ax in range(chart.ndim):
            pad = (chart.coarse_size - 1) / 2
            if int(pad) != pad:
                raise ValueError("`coarse_size` must be odd")
            pad = int(pad)
            if chart.fine_strategy == "jump":
                stride = 1
            elif chart.fine_strategy == "extend":
                stride = chart.fine_size / 2
                if int(stride) != stride:
                    raise ValueError("`fine_size` must be even")
                stride = int(stride)
            else:
                raise AssertionError()
            if ax in chart.irregular_axes:
                pixel_indices.append(
                    jnp.arange(pad, shape_lvl[ax] - pad, stride)
                )
            else:
                pixel_indices.append(jnp.array([pad]))
        pixel_indices = jnp.stack(
            jnp.meshgrid(*pixel_indices, indexing="ij"), axis=-1
        )
        shape_filtered_lvl = pixel_indices.shape[:-1]
        pixel_indices = pixel_indices.reshape(-1, chart.ndim)

        olf, ks = olf_at(lvl, pixel_indices)
        shape_bc_lvl = tuple(
            shape_filtered_lvl[i] if i in chart.irregular_axes else 1
            for i in range(chart.ndim)
        )
        opt_lin_filter.append(olf.reshape(shape_bc_lvl + olf.shape[-2:]))
        kernel_sqrt.append(ks.reshape(shape_bc_lvl + ks.shape[-2:]))

    return RefinementMatrices(opt_lin_filter, kernel_sqrt, cov_sqrt0)
