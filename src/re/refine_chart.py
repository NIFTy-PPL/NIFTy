#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from collections import namedtuple
from functools import partial
from typing import Callable, Iterable, Literal, Optional, Tuple, Union

from jax import numpy as jnp
from jax import vmap
import numpy as np

from .model import AbstractModel
from .refine import _get_cov_from_loc, refine
from .refine_util import (
    coarse2fine_distances,
    coarse2fine_shape,
    fine2coarse_distances,
    fine2coarse_shape,
    get_refinement_shapewithdtype,
)

DEPTH_RANGE = (0, 32)
MAX_SIZE0 = 1024


class CoordinateChart():
    def __init__(
        self,
        min_shape: Optional[Iterable[int]] = None,
        depth: Optional[int] = None,
        *,
        shape0: Optional[Iterable[int]] = None,
        _coarse_size: int = 5,
        _fine_size: int = 4,
        _fine_strategy: Literal["jump", "extend"] = "extend",
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
        """Initialize a refinement chart.

        Parameters
        ----------
        min_shape :
            Minimal extent in pixels along each axes at the final refinement
            level.
        depth :
            Number of refinement iterations.
        shape0 :
            Alternative to `min_shape` and specifies the extent in pixels along
            each axes at the zeroth refinement level.
        _coarse_size :
            Number of coarse pixels which to refine to `_fine_size` fine
            pixels.
        _fine_size :
            Number of fine pixels which to refine from `_coarse_size` coarse
            pixels.
        _fine_strategy :
            Whether to space fine pixels solely within the centermost coarse
            pixel ("jump"), or whether to always space them out s.t. each fine
            pixels takes up half the Euclidean volume of a coarse pixel
            ("extend").
        rg2cart :
            Function to translate Euclidean points on a regular coordinate
            system to the Cartesian coordinate system of the modeled points.
        cart2rg :
            Inverse of `rg2cart`.
        regular_axes :
            Informs the coordinate chart on symmetries within the Cartesian
            coordinate system of the modeled points. If specified, refinement
            matrices are broadcasted as need instead of recomputed.
        irregular_axes :
            Negative of `regular_axes`. Specifying either is sufficient.
        distances :
            Special case of a coordinate chart in which the regular grid points
            are merely stretched or compressed. `distances` are used to set the
            distance between points along every axes at the final refinement
            level.
        distances0:
            Same as `distances` except that `distances0` refers to the
            distances along every axes at the zeroth refinement level.

        Note
        ----
        The functions `rg2cart` and `cart2rg` are always w.r.t. the grid at
        zero depth. In other words, it is straight forward to increase the
        resolution of an existing chart by simply increasing its depth.
        However, extending a grid spatially is more cumbersome and is best done
        via `shape0`.
        """
        if depth is None:
            if min_shape is None:
                raise ValueError("specify `min_shape` to infer `depth`")
            if shape0 is not None or distances0 is not None:
                ve = "can not infer `depth` with `shape0` or `distances0` set"
                raise ValueError(ve)
            for depth in range(*DEPTH_RANGE):
                shape0 = fine2coarse_shape(
                    min_shape,
                    depth=depth,
                    ceil_sizes=True,
                    _coarse_size=_coarse_size,
                    _fine_size=_fine_size,
                    _fine_strategy=_fine_strategy
                )
                if np.prod(shape0, dtype=int) <= MAX_SIZE0:
                    break
            else:
                ve = f"unable to find suitable `depth`; please specify manually"
                raise ValueError(ve)
        if depth < 0:
            raise ValueError(f"invalid `depth`; got {depth!r}")
        self._depth = depth

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
        self._shape0 = (shape0, ) if isinstance(shape0, int) else tuple(shape0)
        self._shape = coarse2fine_shape(
            shape0,
            depth,
            _coarse_size=_coarse_size,
            _fine_size=_fine_size,
            _fine_strategy=_fine_strategy
        )

        if _fine_strategy not in ("jump", "extend"):
            ve = f"invalid `_fine_strategy`; got {_fine_strategy}"
            raise ValueError(ve)

        self._shape_at = partial(
            coarse2fine_shape,
            self.shape0,
            _coarse_size=_coarse_size,
            _fine_size=_fine_size,
            _fine_strategy=_fine_strategy
        )

        self._coarse_size = int(_coarse_size)
        self._fine_size = int(_fine_size)
        self._fine_strategy = _fine_strategy

        # Derived attributes
        self._ndim = len(self.shape)
        self._size = np.prod(self.shape, dtype=int)

        if rg2cart is None and cart2rg is None:
            if distances0 is None and distances is None:
                distances = jnp.ones((self.ndim, ))
                distances0 = fine2coarse_distances(
                    distances,
                    depth,
                    _fine_size=_fine_size,
                    _fine_strategy=_fine_strategy
                )
            elif distances0 is not None:
                distances0 = jnp.broadcast_to(
                    jnp.atleast_1d(distances0), (self.ndim, )
                )
                distances = coarse2fine_distances(
                    distances0,
                    depth,
                    _fine_size=_fine_size,
                    _fine_strategy=_fine_strategy
                )
            else:
                distances = jnp.broadcast_to(
                    jnp.atleast_1d(distances), (self.ndim, )
                )
                distances0 = fine2coarse_distances(
                    distances,
                    depth,
                    _fine_size=_fine_size,
                    _fine_strategy=_fine_strategy
                )

            def _rg2cart(x):
                x = jnp.asarray(x)
                return x * distances0.reshape((-1, ) + (1, ) * (x.ndim - 1))

            def _cart2rg(x):
                x = jnp.asarray(x)
                return x / distances0.reshape((-1, ) + (1, ) * (x.ndim - 1))

            if regular_axes is None and irregular_axes is None:
                regular_axes = tuple(range(self.ndim))
            self._rg2cart = _rg2cart
            self._cart2rg = _cart2rg
        elif rg2cart is not None and cart2rg is not None:
            c0 = jnp.mgrid[tuple(slice(s) for s in self.shape0)]
            if not all(
                jnp.allclose(r, c) for r, c in zip(cart2rg(rg2cart(c0)), c0)
            ):
                raise ValueError("`cart2rg` is not the inverse of `rg2cart`")
            self._rg2cart = rg2cart
            self._cart2rg = cart2rg
            distances = distances0 = None
        else:
            ve = "invalid combination of `cart2rg`, `rg2cart` and `distances`"
            raise ValueError(ve)
        self.distances = distances
        self.distances0 = distances0

        self.distances_at = partial(
            coarse2fine_distances,
            self.distances0,
            _fine_size=_fine_size,
            _fine_strategy=_fine_strategy
        )

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
        self._regular_axes = tuple(regular_axes)
        self._irregular_axes = tuple(irregular_axes)
        if len(self.regular_axes) + len(self.irregular_axes) != self.ndim:
            ve = (
                f"length of regular_axes and irregular_axes"
                f" ({len(self.regular_axes)} + {len(self.irregular_axes)} respectively)"
                f" incompatible with overall dimension {self.ndim}"
            )
            raise ValueError(ve)

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
        self._descr["regular_axes"] = self.regular_axes

    @property
    def shape(self):
        """Shape at the final refinement level"""
        return self._shape

    @property
    def shape0(self):
        """Shape at the zeroth refinement level"""
        return self._shape0

    @property
    def size(self):
        return self._size

    @property
    def ndim(self):
        return self._ndim

    @property
    def depth(self):
        return self._depth

    @property
    def coarse_size(self):
        return self._coarse_size

    @property
    def fine_size(self):
        return self._fine_size

    @property
    def fine_strategy(self):
        return self._fine_strategy

    @property
    def regular_axes(self):
        return self._regular_axes

    @property
    def irregular_axes(self):
        return self._irregular_axes

    def rg2cart(self, positions):
        """Translates positions from the regular Euclidean coordinate system to
        the (in general) irregular Cartesian coordinate system.

        Parameters
        ----------
        positions :
            Positions on a regular Euclidean coordinate system.

        Returns
        -------
        positions :
            Positions on an (in general) irregular Cartesian coordinate system.

        Note
        ----
        This method is independent of the refinement level!
        """
        return self._rg2cart(positions)

    def cart2rg(self, positions):
        """Translates positions from the (in general) irregular Cartesian
        coordinate system to the regular Euclidean coordinate system.

        Parameters
        ----------
        positions :
            Positions on an (in general) irregular Cartesian coordinate system.

        Returns
        -------
        positions :
            Positions on a regular Euclidean coordinate system.

        Note
        ----
        This method is independent of the refinement level!
        """
        return self._cart2rg(positions)

    def rgoffset(self, lvl: int) -> Tuple[float]:
        """Calculate the offset on the regular Euclidean grid due to shrinking
        of the grid with increasing refinement level.

        Parameters
        ----------
        lvl :
            Level of the refinement.

        Returns
        -------
        offset :
            The offset on the regular Euclidean grid along each axes.

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
        """Converts pixel indices to a continuous regular Euclidean grid
        coordinates.

        Parameters
        ----------
        indices :
            Indices of shape `(n_dim, n_indices)` into the NDArray at
            refinement level `lvl` which to convert to points in our regular
            Euclidean grid.
        lvl :
            Level of the refinement.

        Returns
        -------
        rg :
            Regular Euclidean grid coordinates of shape `(n_dim, n_indices)`.
        """
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
        """Converts continuous regular grid positions to pixel indices.

        Parameters
        ----------
        positions :
            Positions on the regular Euclidean coordinate system of shape
            `(n_dim, n_indices)` at refinement level `lvl` which to convert to
            indices in a NDArray at the refinement level `lvl`.
        lvl :
            Level of the refinement.
        discretize :
            Whether to round indices to the next closest integer.

        Returns
        -------
        indices :
            Indices into the NDArray at refinement level `lvl`.
        """
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

        Parameters
        ----------
        indices :
            Indices of shape `(n_dim, n_indices)` into the NDArray at
            refinement level `lvl` which to convert to locations in our (in
            general) irregular coordinate system of the modeled points.
        lvl :
            Level of the refinement.

        Returns
        -------
        positions :
            Positions in the (in general) irregular coordinate system of the
            modeled points of shape `(n_dim, n_indices)`.
        """
        return self.rg2cart(self.ind2rg(indices, lvl))

    def cart2ind(self, positions, lvl, discretize=True):
        """Computes the indices of a pixel given the Cartesian coordinates of
        it.

        Parameters
        ----------
        positions :
            Positions on the Cartesian (in general) irregular coordinate system
            of the modeled points of shape `(n_dim, n_indices)` at refinement
            level `lvl` which to convert to indices in a NDArray at the
            refinement level `lvl`.
        lvl :
            Level of the refinement.
        discretize :
            Whether to round indices to the next closest integer.

        Returns
        -------
        indices :
            Indices into the NDArray at refinement level `lvl`.
        """
        return self.rg2ind(self.cart2rg(positions), lvl, discretize=discretize)

    def shape_at(self, lvl):
        """Retrieves the shape at a given refinement level `lvl`."""
        return self._shape_at(lvl)

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

    def __repr__(self):
        return f"{self.__class__.__name__}(**{self._descr})"

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))


RefinementMatrices = namedtuple(
    "RefinementMatrices", ("filter", "propagator_sqrt", "cov_sqrt0")
)


class RefinementField(AbstractModel):
    def __init__(
        self,
        *args,
        kernel: Optional[Callable] = None,
        dtype=None,
        skip0: bool = False,
        **kwargs
    ):
        """Initialize an Iterative Charted Refinement (ICR) field.

        There are multiple ways to initialize a charted refinement field. The
        recommended way is to first instantiate a `CoordinateChart` and pass it
        as first argument to this method. Alternatively, you may pass any and
        all arguments of `CoordinateChart` also to this method and it will
        instantiate the `CoordinateChart` for you and use it in the same way as
        if directly specified.

        Parameters
        ----------
        chart : CoordinateChart
            The `CoordinateChart` with which to refine.
        kernel :
            Covariance kernel of the refinement field.
        dtype :
            Data-type of the excitations which to add during refining.
        skip0 :
            Whether to skip the first refinement level. This is useful to e.g.
            stack multiple refinement fields on top of each other.
        **kwargs :
            Alternatively to `chart` any parameters accepted by
            `CoordinateChart`.
        """
        self._kernel = kernel
        self._dtype = dtype
        self._skip0 = skip0

        if len(args) > 0 and isinstance(args[0], CoordinateChart):
            if kwargs:
                raise TypeError(f"expected no keyword arguments, got {kwargs}")

            if len(args) == 1:
                self._chart, = args
            elif len(args) == 2 and callable(args[1]) and kernel is None:
                self._chart, self._kernel = args
            elif len(args) == 3 and callable(
                args[1]
            ) and kernel is None and dtype is None:
                self._chart, self._kernel, self._dtype = args
            elif len(args) == 4 and callable(
                args[1]
            ) and kernel is None and dtype is None and skip0 == False:
                self._chart, self._kernel, self._dtype, self._skip0 = args
            else:
                te = "got unexpected arguments in addition to CoordinateChart"
                raise TypeError(te)
        else:
            self._chart = CoordinateChart(*args, **kwargs)

    @property
    def kernel(self):
        """Yields the kernel specified during initialization or throw a
        `TypeError`.
        """
        if self._kernel is None:
            te = (
                "either specify a fixed kernel during initialization of the"
                f" {self.__class__.__name__} class or provide one here"
            )
            raise TypeError(te)
        return self._kernel

    @property
    def dtype(self):
        """Yields the data-type of the excitations."""
        return jnp.float64 if self._dtype is None else self._dtype

    @property
    def skip0(self):
        """Whether to skip the zeroth refinement"""
        return self._skip0

    @property
    def chart(self):
        """Associated `CoordinateChart` with which to iterative refine."""
        return self._chart

    def matrices(
        self,
        kernel: Optional[Callable] = None,
        depth: Optional[int] = None,
        skip0: Optional[bool] = None,
        **kwargs
    ) -> RefinementMatrices:
        """Computes the refinement matrices namely the optimal linear filter
        and the square root of the information propagator (a.k.a. the square
        root of the fine covariance matrix for the excitations) for all
        refinement levels and all pixel indices in the coordinate chart.

        Parameters
        ----------
        kernel :
            Covariance kernel of the refinement field if not specified during
            initialization.
        depth :
            Maximum refinement depth if different to the one of the `CoordinateChart`.
        skip0 :
            Whether to skip the first refinement level.
        """
        kernel = self.kernel if kernel is None else kernel
        depth = self.chart.depth if depth is None else depth
        skip0 = self.skip0 if skip0 is None else skip0

        return _coordinate_refinement_matrices(
            self.chart, kernel=kernel, depth=depth, skip0=skip0, **kwargs
        )

    def matrices_at(
        self,
        level: int,
        pixel_index: Optional[Iterable[int]] = None,
        kernel: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Computes the refinement matrices namely the optimal linear filter
        and the square root of the information propagator (a.k.a. the square
        root of the fine covariance matrix for the excitations) at the
        specified level and pixel index.

        Parameters
        ----------
        level :
            Refinement level.
        pixel_index :
            Index of the NDArray at the refinement level `level` which to
            refine, i.e. use as center coarse pixel.
        kernel :
            Covariance kernel of the refinement field if not specified during
            initialization.
        """
        kernel = self.kernel if kernel is None else kernel

        return _coordinate_pixel_refinement_matrices(
            self.chart,
            level=level,
            pixel_index=pixel_index,
            kernel=kernel,
            **kwargs
        )

    @property
    def domain(self):
        """Yields the `ShapeWithDtype` of the primals."""
        return get_refinement_shapewithdtype(
            shape0=self.chart.shape0,
            depth=self.chart.depth,
            dtype=self.dtype,
            skip0=self.skip0,
            _coarse_size=self.chart.coarse_size,
            _fine_size=self.chart.fine_size,
            _fine_strategy=self.chart.fine_strategy,
        )

    @staticmethod
    def apply(
        xi,
        chart,
        kernel: Union[Callable, RefinementMatrices],
        *,
        skip0: bool = False,
        depth: Optional[int] = None,
        coerce_fine_kernel: bool = True,
        _refine: Optional[Callable] = None,
        precision=None,
    ):
        """Static method to apply a refinement field given some excitations, a
        chart and a kernel.

        Parameters
        ----------
        xi :
            Latent parameters which to use for refining.
        chart :
            Chart with which to refine.
        kernel :
            Covariance kernel with which to build the refinement matrices.
        skip0 :
            Whether to skip the first refinement level.
        depth :
            Refinement depth if different to the depth of the coordinate chart.
        coerce_fine_kernel :
            Whether to coerce the refinement matrices at scales at which the
            kernel matrix becomes singular or numerically highly unstable.
        precision :
            See JAX's precision.
        """
        depth = chart.depth if depth is None else depth
        if depth != len(xi) - 1:
            ve = (
                f"incompatible refinement depths of `xi` ({len(xi) - 1})"
                f" and `depth` (of chart) {depth}"
            )
            raise ValueError(ve)

        if isinstance(kernel, RefinementMatrices):
            refinement = kernel
        else:
            refinement = _coordinate_refinement_matrices(
                chart,
                kernel=kernel,
                depth=depth,
                skip0=skip0,
                coerce_fine_kernel=coerce_fine_kernel
            )
        refine_w_chart = partial(
            refine if _refine is None else _refine,
            _coarse_size=chart.coarse_size,
            _fine_size=chart.fine_size,
            _fine_strategy=chart.fine_strategy,
            precision=precision
        )

        if not skip0:
            fine = (refinement.cov_sqrt0 @ xi[0].ravel()).reshape(xi[0].shape)
        else:
            if refinement.cov_sqrt0 is not None:
                raise AssertionError()
            fine = xi[0]
        for x, olf, k in zip(
            xi[1:], refinement.filter, refinement.propagator_sqrt
        ):
            fine = refine_w_chart(fine, x, olf, k)
        return fine

    def __call__(self, xi, kernel=None, *, skip0=None, **kwargs):
        """See `RefinementField.apply`."""
        kernel = self.kernel if kernel is None else kernel
        skip0 = self.skip0 if skip0 is None else skip0
        return self.apply(xi, self.chart, kernel=kernel, skip0=skip0, **kwargs)

    def __repr__(self):
        descr = f"{self.__class__.__name__}({self.chart!r}"
        descr += f", kernel={self._kernel!r}" if self._kernel is not None else ""
        descr += f", dtype={self._dtype!r}" if self._dtype is not None else ""
        descr += f", skip0={self.skip0!r}" if self.skip0 is not False else ""
        descr += ")"
        return descr

    def __eq__(self, other):
        return repr(self) == repr(other)


def _coordinate_pixel_refinement_matrices(
    chart: CoordinateChart,
    level: int,
    pixel_index: Optional[Iterable[int]] = None,
    kernel: Optional[Callable] = None,
    *,
    coerce_fine_kernel: bool = True,
    _cov_from_loc: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    if pixel_index.size != ndim:
        ve = f"`pixel_index` has {pixel_index.size} dimensions but `chart` has {ndim}"
        raise ValueError(ve)

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
    if coerce_fine_kernel:
        # TODO: Try to work with NaN to avoid the expensive eigendecomposition;
        # work with nan_to_num?
        # Implicitly assume a white power spectrum beyond the numerics limit.
        # Use the diagonal as estimate for the magnitude of the variance.
        fine_kernel_fallback = jnp.diag(jnp.abs(jnp.diag(fine_kernel)))
        # Never produce NaNs (https://github.com/google/jax/issues/1052)
        # This is expensive but necessary (worse but cheaper:
        # `jnp.all(jnp.diag(fine_kernel) > 0.)`)
        is_pos_def = jnp.all(jnp.linalg.eigvalsh(fine_kernel) > 0)
        fine_kernel = jnp.where(is_pos_def, fine_kernel, fine_kernel_fallback)
        # NOTE, subsequently use the Cholesky decomposition, even though
        # already having computed the eigenvalues, as to get consistent results
        # across platforms
    # Matrices are symmetrized by JAX, i.e. gradients are projected to the
    # subspace of symmetric matrices (see
    # https://github.com/google/jax/issues/10815)
    fine_kernel_sqrt = jnp.linalg.cholesky(fine_kernel)

    return olf, fine_kernel_sqrt


def _coordinate_refinement_matrices(
    chart: CoordinateChart,
    kernel: Callable,
    *,
    depth: Optional[int] = None,
    skip0=False,
    coerce_fine_kernel: bool = True,
    _cov_from_loc=None
) -> RefinementMatrices:
    cov_from_loc = _get_cov_from_loc(kernel, _cov_from_loc)
    depth = chart.depth if depth is None else depth

    if not skip0:
        rg0 = jnp.mgrid[tuple(slice(s) for s in chart.shape0)]
        c0 = jnp.stack(chart.ind2cart(rg0, 0), axis=-1).reshape(-1, chart.ndim)
        # Matrices are symmetrized by JAX, i.e. gradients are projected to the
        # subspace of symmetric matrices (see
        # https://github.com/google/jax/issues/10815)
        cov_sqrt0 = jnp.linalg.cholesky(cov_from_loc(c0, c0))
    else:
        cov_sqrt0 = None

    opt_lin_filter, kernel_sqrt = [], []
    olf_at = vmap(
        partial(
            _coordinate_pixel_refinement_matrices,
            chart,
            coerce_fine_kernel=coerce_fine_kernel,
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
