from dataclasses import field

import jax.numpy as jnp

from ..model import Model
from ..prior import NormalPrior
from ..tree_math import ShapeWithDtype
from .kernel import ICRKernel, apply_kernel


class ICRCorrelatedField(Model):
    offset: Model = field(metadata=dict(static=False))
    covariance: Model = field(metadata=dict(static=False))

    def __init__(
        self,
        grid,
        *,
        kernel=dict(kind="matern"),
        offset=0.0,
        window_size=None,
        compress=dict(rtol=1e-5, atol=1e-10, buffer_size=10_000, use_distances=True),
        prefix="mgcfm",
    ):
        from .matern import MaternCovariance

        self.grid = grid
        shapes = [
            ShapeWithDtype(self.grid.at(lvl).shape, jnp.float_)
            for lvl in range(grid.depth + 1)
        ]
        self._name_exc = str(prefix) + "excitations"
        domain = {self._name_exc: shapes}

        DEFAULT_KERNEL_KIND = "matern"
        if not isinstance(kernel, dict):
            raise TypeError(f"invalid kernel type; got {kernel!r}")
        fixed_kernel = False
        kernel = kernel.copy()
        kernel_kind = kernel.pop("kind", DEFAULT_KERNEL_KIND).lower()
        if kernel_kind == "matern":
            covariance = MaternCovariance(**kernel, prefix=prefix)
        elif kernel_kind == "fixed":
            fixed_kernel = True
            covariance = kernel["covariance"]
        else:
            raise ValueError(f"kernel {kernel_kind!r} not supported")
        self.covariance = covariance
        self.fixed_kernel = fixed_kernel
        if not self.fixed_kernel:
            domain |= self.covariance.domain

        if isinstance(offset, (tuple, list)):
            offset = NormalPrior(*offset, name=prefix + "offset")
        elif callable(offset) or not isinstance(offset, float):
            raise ValueError
        self.offset = offset
        if callable(self.offset):
            domain |= self.offset.domain

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
        off = self.offset(x) if callable(self.offset) else self.offset
        return off + apply_kernel(x[self._name_exc], kernel=kernel)[-1]
