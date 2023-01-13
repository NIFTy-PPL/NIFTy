#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial

import healpy as hp
import jax
from jax import numpy as jnp
from jax import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import nifty8.re as jft

jax.config.update("jax_debug_nans", True)


# %%
def matern_kernel(distance, scale, cutoff, dof):
    """Evaluates the Matern covariance kernel parametrized by its `scale`,
    length scale (a.k.a. `cutoff`) and degree-of-freedom parameter `dof` at
    `distance`.
    """
    if dof == 0.5:
        cov = scale**2 * jnp.exp(-distance / cutoff)
    elif dof == 1.5:
        reg_dist = jnp.sqrt(3) * distance / cutoff
        cov = scale**2 * (1 + reg_dist) * jnp.exp(-reg_dist)
    elif dof == 2.5:
        reg_dist = jnp.sqrt(5) * distance / cutoff
        cov = scale**2 * (1 + reg_dist + reg_dist**2 / 3) * jnp.exp(-reg_dist)
    else:
        from jax.scipy.special import gammaln
        from scipy.special import kv
        from warnings import warn

        warn("falling back to generic Matern covariance function")
        reg_dist = jnp.sqrt(2 * dof) * distance / cutoff
        cov = scale**2 * 2**(1 - dof) / jnp.exp(
            gammaln(dof)
        ) * (reg_dist)**dof * kv(dof, reg_dist)

    # NOTE, this is not safe for differentiating because `cov` still may
    # contain NaNs
    return jnp.where(distance < 1e-8 * cutoff, scale**2, cov)


# %%
nest = True
kernel = partial(matern_kernel, scale=1., cutoff=1., dof=1.5)

def rg2cart(x, idx0, scl):
    """Transforms regular, points from a Euclidean space to irregular points in
    an cartesian coordinate system in 1D."""
    return jnp.exp(scl * x[0] + idx0)[np.newaxis, ...]


def cart2rg(x, idx0, scl):
    """Inverse of `rg2cart`."""
    return ((jnp.log(x[0]) - idx0) / scl)[np.newaxis, ...]


# %%
cc = jft.HEALPixChart(
    min_shape=(12 * 128**2, 4, ),
    nonhp_rg2cart=partial(rg2cart, idx0=-0.27, scl=1.1),
    nonhp_cart2rg=partial(cart2rg, idx0=-0.27, scl=1.1),
    _coarse_size=3,
    _fine_size=2,
)
rf = jft.RefinementHPField(cc)

key = random.PRNGKey(42)
xi = jft.random_like(key, rf.domain)
refined = rf(xi, kernel)

# %%
for i in range(refined.shape[1]):
    hp.mollview(rf(xi[:1], kernel)[:, i], nest=nest)
    hp.mollview(refined[:, i], nest=nest)
plt.show()
