#!/usr/bin/env python3
import jax
from jax import random
import dataclasses
from typing import Any

import nifty8.re as jft

jax.config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)
key, subkey1, subkey2, noise_key = random.split(key, 4)

dims = (128, 128)

cf_zm = dict(offset_mean=0.0, offset_std=(1e-3, 1e-4))
cf_fl = dict(
    fluctuations=(1e-1, 5e-3),
    loglogavgslope=(-1.0, 1e-2),
    flexibility=(1e0, 5e-1),
    asperity=(5e-1, 5e-2),
)
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(
    dims, distances=1.0 / dims[0], **cf_fl, prefix="ax1", non_parametric_kind="power"
)
correlated_field = cfm.finalize()

const = random.normal(subkey1, dims)
mock = jft.random_like(subkey2, correlated_field.domain)

class Response(jft.Model):
    constant: Any = dataclasses.field(metadata=dict(static=False))
    def __init__(self, constant):
        self.constant = constant
        super().__init__(domain=jft.ShapeWithDtype(dims))

    def __call__(self, x):
        return x * self.constant


response = Response(const)
print("constant", const)
print("leaves", jax.jax.tree.leaves(response))

data = response(correlated_field(mock)) + jft.random_like(noise_key,
                                                          correlated_field.target)

llh = jft.Gaussian(data).amend(response)
print("leaves of likelihood without signal model", jax.jax.tree.leaves(llh))

full_lh = llh.amend(correlated_field) # Here the "const" array is missing
print("leaves of likelihood with signal model", jax.jax.tree.leaves(full_lh))
