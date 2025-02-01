#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from dataclasses import field
from typing import Callable, Union

import jax.numpy as jnp
from jax import eval_shape
from jax.tree_util import Partial
from numpy import typing as npt

from ..model import Model, WrappedCall
from ..prior import NormalPrior
from ..tree_math import ShapeWithDtype
from .grid import Grid
from .kernel import ICRKernel, Kernel, apply_kernel


class ICRField(Model):
    grid: Grid
    kernel: Kernel
    covariance: Union[Model, Callable] = field(metadata=dict(static=False))
    offset: Model = field(metadata=dict(static=False))
    compress: bool
    fixed_kernel: bool

    def __init__(
        self,
        grid: Grid,
        *,
        kernel: Union[
            dict, Model, Callable[[npt.NDArray, npt.NDArray], npt.NDArray]
        ] = dict(kind="experimental_matern"),
        offset=0.0,
        window_size=None,
        compress: Union[bool, dict] = dict(
            rtol=1e-5, atol=1e-10, buffer_size=10_000, use_distances=True
        ),
        prefix="mgcfm",
    ):
        """Correlated field model utilizing ICR to work on arbitrarily charted grids.

        Parameters
        ----------
        grid: Grid
            Storage object with instructions on how pixels are spaced.
        kernel: dict or Model or callable
            Parameters for the kernel. Currently supports `kind='experimental_matern'`
            with the following additional arguments

            - scale: tuple or Model or float -- Prior scale of the kernel.
            - cutoff: tuple or Model or float -- Prior cutoff of the covariance kernel.
            - loglogslope: tuple or Model or float -- Prior power-law slope of the covariance kernel correlating modes.
            - n_integrate = 2_000 -- Number of integration points for the harmonic transformation.
            - n_interpolate = 512 -- Number of interpolation points for interpolating the kernel.
            - interpolation_dists_min_max -- Interpolation range. Automatically chosen based on cutoff if left unspecified.
            - integration_dists_min_max -- Integration range for the harmonic transform. Automatically chosen based on cutoff if left unspecified.

            alternatively, `kernel` can be a NIFTy `Model` which yields a callable
            covariance function (`kernel(xi: dict) -> callable[[x, y], z]`), or a
            `callable[[x, y], z]` covarinace function.
        offset: tuple or callable or float
            Prior shift from zero in addition to the field intrinsic random shift.
        window_size:
            Shape of the refinement matrices. Its value is automatically inferred
            from the padding in the grid or set to sane defaults if left unspecified.
        compress: dict
            Arguments for compressing the refinement matrices. If this is set to an
            empty dictionary, no compression is applied.
        prexi: str
            Prefix for the latent parameter names.
        """
        from .matern import MaternHarmonicCovariance

        self.grid = grid
        shapes = [
            ShapeWithDtype(self.grid.at(lvl).shape, jnp.float_)
            for lvl in range(grid.depth + 1)
        ]
        self._name_exc = str(prefix) + "excitations"
        domain = {self._name_exc: shapes}

        # Parse kernel
        DEFAULT_KERNEL_KIND = "experimental_matern"
        fixed_kernel, covariance = False, None
        if isinstance(kernel, dict):
            kernel = kernel.copy()
            kernel_kind = kernel.pop("kind", DEFAULT_KERNEL_KIND).lower()
            if kernel_kind == "experimental_matern":
                coord_swd = eval_shape(
                    grid.at(0).index2coord, ShapeWithDtype((grid.at(0).ndim,), jnp.int_)
                )
                kernel.setdefault("ndim", coord_swd.shape[0])
                covariance = MaternHarmonicCovariance(**kernel, prefix=prefix)
            else:
                raise ValueError(f"kernel {kernel_kind!r} not supported")
        elif isinstance(kernel, Model):
            covariance = kernel
        elif callable(kernel):
            fixed_kernel = True
            covariance = Partial(kernel)
        else:
            raise TypeError(f"invalid kernel type; got {kernel!r}")
        self.fixed_kernel = fixed_kernel
        self.covariance = covariance
        if not self.fixed_kernel:
            domain |= self.covariance.domain

        # Parse offset
        name_off = prefix + "offset"
        if isinstance(offset, (tuple, list)):
            offset = NormalPrior(*offset, name=name_off)
        elif callable(offset) and not isinstance(offset, Model):
            offset = WrappedCall(offset, name=name_off, white_init=True)
        if not (isinstance(offset, Model) or isinstance(offset, float)):
            raise ValueError(f"invalid `offset`; got {offset!r}")
        if isinstance(offset, Model):
            domain |= offset.domain
        self.offset = offset

        self.compress = isinstance(compress, dict) and len(compress) > 0
        kernel = ICRKernel(self.grid, None, window_size=window_size)
        if self.compress:
            kernel = kernel.compress_indices(**compress)
        if self.fixed_kernel:
            kernel = kernel.replace(covariance=self.covariance)
            if self.compress:
                kernel = kernel.compress_matrices()
        self.kernel = kernel

        # TODO: join init instead of domain
        super().__init__(domain=domain, white_init=True)

    def __call__(self, x):
        if not self.fixed_kernel:
            kernel = self.kernel.replace(covariance=self.covariance(x))
            if self.compress:
                kernel = kernel.compress_matrices()
        off = self.offset(x) if isinstance(self.offset, Model) else self.offset
        return off + apply_kernel(x[self._name_exc], kernel=kernel)[-1]
