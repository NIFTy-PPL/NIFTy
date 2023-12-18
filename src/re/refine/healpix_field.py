#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
from typing import Callable, Optional, Union

import jax
import numpy as np
from jax import numpy as jnp

from ..tree_math import ShapeWithDtype
from ..logger import logger
from ..model import LazyModel
from ..num import amend_unique_
from .chart import HEALPixChart
from .healpix_refine import cov_sqrt as cov_sqrt_hp
from .healpix_refine import refine as refine_hp
from .util import (
    RefinementMatrices, get_cov_from_loc, get_refinement_shapewithdtype,
    refinement_matrices
)


def _matrices_naive(
    chart,
    kernel: Callable,
    depth: int,
    *,
    coerce_fine_kernel: bool = False,
):
    cov_from_loc = get_cov_from_loc(kernel)

    def cov_mat(lvl, idx_hp, idx_r):
        # `idx_r` is the left-most radial pixel of the to-be-refined slice
        # Extend `gc` and `gf` radially
        gc, gf = chart.get_coarse_fine_pair((idx_hp, idx_r), lvl)
        assert gf.shape[0] == chart.fine_size**(chart.ndim + 1)

        coord = jnp.concatenate((gc, gf), axis=0)
        return cov_from_loc(coord, coord)

    def ref_mat_from_cov(cov):
        olf, ks = refinement_matrices(
            cov,
            chart.fine_size**(chart.ndim + 1),
            coerce_fine_kernel=coerce_fine_kernel
        )
        if chart.ndim > 1:
            olf = olf.reshape(
                chart.fine_size**2, chart.fine_size, chart.coarse_size**2,
                chart.coarse_size
            )
        return olf, ks

    def ref_mat(lvl, idx_hp, idx_r):
        return ref_mat_from_cov(cov_mat(lvl, idx_hp, idx_r))

    opt_lin_filter, kernel_sqrt = [], []
    for lvl in range(depth):
        pix_hp_idx = jnp.arange(chart.shape_at(lvl)[0])
        if chart.ndim == 1:
            pix_r_off = None
            vdist = jax.vmap(partial(ref_mat, lvl), in_axes=(0, 0))
        elif chart.ndim == 2:
            pix_r_off = jnp.arange(
                chart.shape_at(lvl)[1] - chart.coarse_size + 1
            )
            vdist = jax.vmap(partial(ref_mat, lvl), in_axes=(None, 0))
            vdist = jax.vmap(vdist, in_axes=(0, None))
        else:
            raise AssertionError()
        olf, ks = vdist(pix_hp_idx, pix_r_off)
        opt_lin_filter.append(olf)
        kernel_sqrt.append(ks)

    return RefinementMatrices(
        opt_lin_filter, kernel_sqrt, None, (None, ) * len(opt_lin_filter)
    )


def _matrices_tol(
    chart,
    kernel: Callable,
    depth: int,
    *,
    coerce_fine_kernel: bool = False,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    which: Optional[str] = "dist",
    mat_buffer_size: Optional[int] = None,
    _verbosity: int = 0,
):
    dist_mat_from_loc = get_cov_from_loc(lambda x: x, None)
    if which not in (None, "dist", "cov"):
        ve = f"expected `which` to be one of 'dist' or 'cov'; got {which!r}"
        raise ValueError(ve)
    if atol is not None and rtol is not None and mat_buffer_size is None:
        raise TypeError("must specify `mat_buffer_size`")

    def dist_mat(lvl, idx_hp, idx_r):
        # `idx_r` is the left-most radial pixel of the to-be-refined slice
        # Extend `gc` and `gf` radially
        gc, gf = chart.get_coarse_fine_pair((idx_hp, idx_r), lvl)
        assert gf.shape[0] == chart.fine_size**(chart.ndim + 1)

        coord = jnp.concatenate((gc, gf), axis=0)
        return dist_mat_from_loc(coord, coord)

    def ref_mat(cov):
        olf, ks = refinement_matrices(
            cov,
            chart.fine_size**(chart.ndim + 1),
            coerce_fine_kernel=coerce_fine_kernel
        )
        if chart.ndim > 1:
            olf = olf.reshape(
                chart.fine_size**2, chart.fine_size, chart.coarse_size**2,
                chart.coarse_size
            )
        return olf, ks

    opt_lin_filter, kernel_sqrt, idx_map = [], [], []
    for lvl in range(depth):
        pix_hp_idx = jnp.arange(chart.shape_at(lvl)[0])
        assert chart.ndim == 2
        pix_r_off = jnp.arange(chart.shape_at(lvl)[1] - chart.coarse_size + 1)
        # Map only over the radial axis b/c that is irregular anyways simply
        # due to the HEALPix geometry and manually scan over the HEALPix axis
        # to only save unique values
        vdist = jax.vmap(partial(dist_mat, lvl), in_axes=(None, 0))

        # Successively amend the duplicate-free distance/covariance matrices
        d = jax.eval_shape(vdist, pix_hp_idx[0], pix_r_off)
        u = jnp.full((mat_buffer_size, ) + d.shape, jnp.nan, dtype=d.dtype)

        def scanned_amend_unique(u, pix):
            d = vdist(pix, pix_r_off)
            d = kernel(d) if which == "cov" else d
            if _verbosity > 1:
                # Magic code to move up curser by one line and delete whole line
                msg = "\x1b[1A\x1b[2K{pix}/{n}"
                jax.debug.print(msg, pix=pix, n=pix_hp_idx[-1])
            return amend_unique_(u, d, axis=0, atol=atol, rtol=rtol)

        if _verbosity > 1:
            jax.debug.print("")
        u, inv = jax.lax.scan(scanned_amend_unique, u, pix_hp_idx)
        # Cut away the placeholder for preserving static shapes
        n = np.unique(inv).size
        if n >= u.shape[0] or not np.all(np.isnan(u[n:])):
            raise ValueError("`mat_buffer_size` too small")
        u = u[:n]
        u = kernel(u) if which == "dist" else u
        inv = np.array(inv)
        if _verbosity > 0:
            logger.info(f"Post uniquifying: {u.shape}")

        # Finally, all distance/covariance matrices are assembled and we
        # can map over them to construct the refinement matrices as usual
        vmat = jax.vmap(jax.vmap(ref_mat, in_axes=(0, )), in_axes=(0, ))
        olf, ks = vmat(u)

        opt_lin_filter.append(olf)
        kernel_sqrt.append(ks)
        idx_map.append(inv)

    return RefinementMatrices(opt_lin_filter, kernel_sqrt, None, idx_map)


class RefinementHPField(LazyModel):
    def __init__(
        self,
        chart: HEALPixChart,
        kernel: Optional[Callable] = None,
        dtype=None,
        skip0: bool = False,
    ):
        """Initialize an Iterative Charted Refinement (ICR) field for a HEALPix
        map with a radial extent.

        Parameters
        ----------
        chart :
            HEALPix coordinate chart with which to iteratively refine.
        kernel :
            Covariance kernel of the refinement field.
        dtype :
            Data-type of the excitations which to add during refining.
        skip0 :
            Whether to skip the first refinement level. This is useful to e.g.
        """
        self._chart = chart
        self._kernel = kernel
        self._dtype = dtype
        self._skip0 = skip0

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
        """Associated `HEALPixChart` with which to iteratively refine."""
        return self._chart

    def matrices(
        self,
        kernel: Optional[Callable] = None,
        depth: Optional[int] = None,
        skip0: Optional[bool] = None,
        *,
        coerce_fine_kernel: bool = False,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        which: Optional[str] = "dist",
        mat_buffer_size: Optional[int] = None,
        _verbosity: int = 0,
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
            Maximum refinement depth if different to the one of the
            `HEALPixChart`.
        skip0 :
            Whether to skip the first refinement level.
        coerce_fine_kernel :
            Whether to coerce the refinement matrices at scales at which the
            kernel matrix becomes singular or numerically highly unstable.
        atol :
            Absolute tolerance of the element-wise difference between distance
            matrices up to which refinement matrices are merged.
        rtol :
            Relative tolerance of the element-wise difference between distance
            matrices up to which refinement matrices are merged.
        which : 'cov' or 'dist'
            Type of the matrix for which the unique instances shall be found.
            If the covariance matrices (`cov`) are "uniquified", then the
            procedure depends on the kernel used. If the distance matrices
            (`dist`) are "uniquified", then the search for redundancies becomes
            independent of the kernel. For a fixed charting, the latter can
            always be done ahead of time while it might be easier to set
            sensible `atol` and `rtol` parameters for the latter.

        Note
        ----
        Finding duplicates in the distance matrices is computationally
        expensive if there are only few of them. However, if the chart is kept
        fixed, then this one time investment might still be worth it though.
        """
        kernel = self.kernel if kernel is None else kernel
        depth = self.chart.depth if depth is None else depth
        skip0 = self.skip0 if skip0 is None else skip0

        if atol is None and rtol is None:
            rfm = _matrices_naive(
                self.chart,
                kernel,
                depth,
                coerce_fine_kernel=coerce_fine_kernel
            )
        else:
            rfm = _matrices_tol(
                self.chart,
                kernel,
                depth,
                coerce_fine_kernel=coerce_fine_kernel,
                atol=atol,
                rtol=rtol,
                which=which,
                mat_buffer_size=mat_buffer_size,
                _verbosity=_verbosity
            )
        cov_sqrt0 = cov_sqrt_hp(self.chart, kernel) if not skip0 else None
        return rfm._replace(cov_sqrt0=cov_sqrt0)

    @property
    def domain(self):
        """Yields the `ShapeWithDtype` of the primals."""
        nonhp_domain = get_refinement_shapewithdtype(
            shape0=self.chart.shape0[1:],
            depth=self.chart.depth,
            dtype=self.dtype,
            skip0=self.skip0,
            _coarse_size=self.chart.coarse_size,
            _fine_size=self.chart.fine_size,
            _fine_strategy=self.chart.fine_strategy,
        )
        if not self.skip0:
            domain = [
                ShapeWithDtype(
                    (12 * self.chart.nside0**2, ) + nonhp_domain[0].shape,
                    nonhp_domain[0].dtype
                )
            ]
        else:
            domain = [None]
        domain += [
            ShapeWithDtype(
                (12 * self.chart.nside_at(lvl)**2, ) + swd.shape[:-1] +
                (4 * swd.shape[-1], ), swd.dtype
            )
            for lvl, swd in zip(range(self.chart.depth + 1), nonhp_domain[1:])
        ]
        return domain

    @staticmethod
    def apply(
        xi,
        chart: HEALPixChart,
        kernel: Union[Callable, RefinementMatrices],
        *,
        skip0: bool = False,
        coerce_fine_kernel: bool = False,
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
            Chart with which to refine the non-HEALPix axis.
        kernel :
            Covariance kernel with which to build the refinement matrices.
        skip0 :
            Whether to skip the first refinement level.
        coerce_fine_kernel :
            Whether to coerce the refinement matrices at scales at which the
            kernel matrix becomes singular or numerically highly unstable.
        precision :
            See JAX's precision.
        """
        if xi[0].shape != chart.shape0:
            ve = "zeroth excitations do not fit to chart"
            raise ValueError(ve)

        if isinstance(kernel, RefinementMatrices):
            refinement = kernel
        else:
            refinement = RefinementHPField(chart, None, xi[0].dtype).matrices(
                kernel=kernel,
                skip0=skip0,
                coerce_fine_kernel=coerce_fine_kernel
            )
        refine_w_chart = partial(
            refine_hp if _refine is None else _refine,
            chart=chart,
            precision=precision
        )

        if not skip0:
            fine = (refinement.cov_sqrt0 @ xi[0].ravel()).reshape(xi[0].shape)
        else:
            if refinement.cov_sqrt0 is not None:
                raise AssertionError()
            fine = xi[0]
        for x, olf, k, im in zip(
            xi[1:], refinement.filter, refinement.propagator_sqrt,
            refinement.index_map
        ):
            fine = refine_w_chart(fine, x, olf, k, im)
        return fine

    def __call__(self, xi, kernel=None, *, skip0=None, **kwargs):
        """See `RefinementField.apply`."""
        kernel = self.kernel if kernel is None else kernel
        skip0 = self.skip0 if skip0 is None else skip0
        return self.apply(xi, self.chart, kernel=kernel, skip0=skip0, **kwargs)

    def __repr__(self):
        descr = f"{self.__class__.__name__}(chart={self.chart!r}"
        descr += f", kernel={self._kernel!r}" if self._kernel is not None else ""
        descr += f", dtype={self._dtype!r}" if self._dtype is not None else ""
        descr += f", skip0={self.skip0!r}" if self.skip0 is not False else ""
        descr += ")"
        return descr

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))
