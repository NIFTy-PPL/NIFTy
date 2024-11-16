import jax.numpy as jnp

from dataclasses import field

from .kernel import KernelBase
from ..correlated_field import CorrelatedFieldMaker
from ..model import Initializer, Model


class CFCov(Model):
    cf: Model = field(metadata=dict(static=False))

    def __init__(self, pre, size, binsize, offset_mean, offset_std, **fluct_kwargs):
        cf = CorrelatedFieldMaker(pre)
        cf.set_amplitude_total_offset(offset_mean, offset_std)
        cf.add_fluctuations((2 * size,), (binsize,), **fluct_kwargs)
        self.cf = cf.finalize()
        self.size = size
        self.binsize = binsize
        super().__init__(domain=self.cf.domain, init=self.cf.init, target=callable)

    def __call__(self, x):
        res = self.cf(x)[: self.size]

        def ker(x, y):
            r = jnp.linalg.norm(x - y)
            r = r / self.binsize
            r = jnp.clip(r, max=self.size * self.binsize)
            if isinstance(r, float):
                r = int(r)
            else:
                r = r.astype(int)
            return res[r]

        return ker


class VariableKernel(Model):
    covariance: Model = field(metadata=dict(static=False))
    kernel: KernelBase = field(metadata=dict(static=False))

    def __init__(self, covariance, kernel, name):
        self.covariance = covariance
        self.kernel = kernel
        self.name = str(name)
        init = self.covariance.init | Initializer(
            {self.name: kernel.init._call_or_struct}
        )
        # TODO make others
        super().__init__(init=init, target=self.kernel.target)

    def __call__(self, x):
        return self.kernel.apply(x[self.name], _covariance=self.covariance(x))
