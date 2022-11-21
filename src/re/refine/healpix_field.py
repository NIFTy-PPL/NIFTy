#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
from typing import Callable, Optional, Tuple, Union

from jax import numpy as jnp
from jax import vmap

from ..forest_util import ShapeWithDtype
from ..model import AbstractModel
from .healpix_refine import refine as refine_hp
from .healpix_refine import cov_sqrt as cov_sqrt_hp
from .chart import HEALPixChart
from .util import (
    RefinementMatrices, get_cov_from_loc, get_refinement_shapewithdtype
)


def _healpix_pixel_refinement_matrices(
    gc_and_gf,
    kernel: Optional[Callable] = None,
    *,
    coerce_fine_kernel: bool = True,
    _cov_from_loc: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    cov_from_loc = get_cov_from_loc(kernel, _cov_from_loc)
    n_fsz = gc_and_gf[1].shape[0]

    coord = jnp.concatenate(gc_and_gf, axis=0)
    del gc_and_gf
    cov = cov_from_loc(coord, coord)
    del coord
    cov_ff = cov[-n_fsz:, -n_fsz:]
    cov_fc = cov[-n_fsz:, :-n_fsz]
    cov_cc = cov[:-n_fsz, :-n_fsz]
    del cov
    cov_cc_inv = jnp.linalg.inv(cov_cc)
    del cov_cc

    olf = cov_fc @ cov_cc_inv
    # Also see Schur-Complement
    fine_kernel = cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
    del cov_cc_inv, cov_fc, cov_ff
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


class RefinementHPField(AbstractModel):
    def __init__(
        self,
        chart: HEALPixChart,
        kernel: Optional[Callable] = None,
        dtype=None,
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
        """
        self._chart = chart
        self._kernel = kernel
        self._dtype = dtype

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
    def chart(self):
        """Associated `HEALPixChart` with which to iteratively refine."""
        return self._chart

    def matrices(
        self,
        kernel: Optional[Callable] = None,
        depth: Optional[int] = None,
        coerce_fine_kernel: bool = True,
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
        """
        cc = self.chart
        kernel = self.kernel if kernel is None else kernel
        depth = cc.depth if depth is None else depth

        def mat(lvl, idx_hp, idx_r):
            # `idx_r` is the left-most radial pixel of the to-be-refined slice
            # Extend `gc` and `gf` radially
            gc, gf = cc.get_coarse_fine_pair((idx_hp, idx_r), lvl)
            olf, ks = _healpix_pixel_refinement_matrices(
                (gc, gf), kernel=kernel, coerce_fine_kernel=coerce_fine_kernel
            )
            if cc.ndim > 1:
                olf = olf.reshape(
                    cc.fine_size**2, cc.fine_size, cc.coarse_size**2,
                    cc.coarse_size
                )
            return olf, ks

        cov_sqrt0 = cov_sqrt_hp(cc, kernel)

        opt_lin_filter, kernel_sqrt = [], []
        for lvl in range(depth):
            pix_hp_idx = jnp.arange(cc.shape_at(lvl)[0])
            if cc.ndim == 1:
                pix_r_off = None
                vmat = vmap(partial(mat, lvl), in_axes=(0, 0))
            elif cc.ndim == 2:
                pix_r_off = jnp.arange(cc.shape_at(lvl)[1] - cc.coarse_size + 1)
                vmat = vmap(partial(mat, lvl), in_axes=(None, 0))
                vmat = vmap(vmat, in_axes=(0, None))
            else:
                raise AssertionError()
            olf, ks = vmat(pix_hp_idx, pix_r_off)
            opt_lin_filter.append(olf)
            kernel_sqrt.append(ks)

        return RefinementMatrices(opt_lin_filter, kernel_sqrt, cov_sqrt0)

    @property
    def domain(self):
        """Yields the `ShapeWithDtype` of the primals."""
        nonhp_domain = get_refinement_shapewithdtype(
            shape0=self.chart.shape0[1:],
            depth=self.chart.depth,
            dtype=self.dtype,
            skip0=False,
            _coarse_size=self.chart.coarse_size,
            _fine_size=self.chart.fine_size,
            _fine_strategy=self.chart.fine_strategy,
        )
        domain = [
            ShapeWithDtype(
                (12 * self.chart.nside0**2, ) + nonhp_domain[0].shape,
                nonhp_domain[0].dtype
            )
        ]
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
        radial_chart :
            Chart with which to refine the radial axis of the HEALPix map.
        kernel :
            Covariance kernel with which to build the refinement matrices.
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
                kernel=kernel, coerce_fine_kernel=coerce_fine_kernel
            )
        refine_w_chart = partial(
            refine_hp if _refine is None else _refine,
            chart=chart,
            precision=precision
        )

        fine = (refinement.cov_sqrt0 @ xi[0].ravel()).reshape(xi[0].shape)
        for x, olf, k in zip(
            xi[1:], refinement.filter, refinement.propagator_sqrt
        ):
            fine = refine_w_chart(fine, x, olf, k)
        return fine

    def __call__(self, xi, kernel=None, **kwargs):
        """See `RefinementField.apply`."""
        kernel = self.kernel if kernel is None else kernel
        return self.apply(xi, self.chart, kernel=kernel, **kwargs)

    def __repr__(self):
        descr = f"{self.__class__.__name__}(chart={self.chart!r}"
        descr += f", kernel={self._kernel!r}" if self._kernel is not None else ""
        descr += f", dtype={self._dtype!r}" if self._dtype is not None else ""
        descr += ")"
        return descr

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))
