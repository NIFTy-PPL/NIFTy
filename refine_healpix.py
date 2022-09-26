#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from collections import namedtuple
from functools import partial
import sys

import healpy as hp
from healpy import pixelfunc
import jax
from jax import numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np

import nifty8.re as jft
from nifty8.re.refine import _get_cov_from_loc

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

pix0s = np.stack(pixelfunc.pix2vec(1, np.arange(12), nest=nest), axis=-1)
cov_from_loc = _get_cov_from_loc(kernel, None)
fks_sqrt = jnp.linalg.cholesky(cov_from_loc(pix0s, pix0s))

key = random.PRNGKey(43)
r0 = random.normal(key, (12, ))
refined = fks_sqrt @ r0
hp.mollview(refined, nest=nest)
depth = 6
key = random.PRNGKey(42)
for i in range(depth):
    _, key = random.split(key)
    exc = random.normal(key, (refined.shape[0], 4))
    refined = jft.refine_healpix.refine(refined, exc, kernel)
    hp.mollview(refined, nest=nest)
# plt.show()

# %%
key = random.PRNGKey(42)


def rg2cart(x, idx0, scl):
    """Transforms regular, points from a Euclidean space to irregular points in
    an cartesian coordinate system in 1D."""
    return jnp.exp(scl * x[0] + idx0)[np.newaxis, ...]


def cart2rg(x, idx0, scl):
    """Inverse of `rg2cart`."""
    return ((jnp.log(x[0]) - idx0) / scl)[np.newaxis, ...]


n_r = 4
radial_chart = jft.CoordinateChart(
    min_shape=(n_r, ),
    depth=1,
    rg2cart=partial(rg2cart, idx0=-0.27, scl=1.1),
    cart2rg=partial(cart2rg, idx0=-0.27, scl=1.1),
    _coarse_size=3,
    _fine_size=2,
)
pix0s = np.stack(pixelfunc.pix2vec(1, np.arange(12), nest=nest), axis=-1)
pix0s = (
    pix0s[:, np.newaxis, :] *
    radial_chart.ind2cart(jnp.arange(n_r)[np.newaxis, :], -1)[..., np.newaxis]
).reshape(12 * n_r, 3)
cov_from_loc = _get_cov_from_loc(kernel, None)
fks_sqrt = jnp.linalg.cholesky(cov_from_loc(pix0s, pix0s))

r0 = random.normal(key, (12 * n_r, ))
refined = (fks_sqrt @ r0).reshape(12, n_r)
depth = 7
for i in range(depth):
    _, key = random.split(key)
    exc = random.normal(key, (refined.shape[0], refined.shape[1] - 2, 8))
    refined = jft.refine_healpix.refine(refined, exc, kernel, radial_chart)

# %%
for i in range(refined.shape[1]):
    hp.mollview((fks_sqrt @ r0).reshape(12, n_r)[:, i], nest=nest)
    hp.mollview(refined[:, i], nest=nest)
# plt.show()

# %%
Timed = namedtuple(
    "Timed",
    ("time", "number", "repeat", "median", "min", "max", "mean", "std"),
    rename=True
)


def timeit(stmt, setup="pass", repeat=7, number=None):
    """Handy timer utility returning the median time it took evaluate `stmt`."""
    import timeit

    timer = timeit.Timer(stmt, setup=setup)
    if number is None:
        number, _ = timer.autorange()
    timings = np.array(timer.repeat(repeat=repeat, number=number)) / number

    t = np.median(timings)
    mi, ma = np.min(timings), np.max(timings)
    m, std = np.mean(timings), np.std(timings)
    return Timed(
        time=t,
        number=number,
        repeat=repeat,
        median=t,
        min=mi,
        max=ma,
        mean=m,
        std=std
    )


pix0s = np.stack(pixelfunc.pix2vec(1, np.arange(12), nest=nest), axis=-1)
cov_from_loc = _get_cov_from_loc(kernel, None)
fks_sqrt = jnp.linalg.cholesky(cov_from_loc(pix0s, pix0s))

key = random.PRNGKey(12)
r0 = random.normal(key, (12, ))
refined = fks_sqrt @ r0
depth = 9
for i in range(depth):
    _r = refined
    exc = random.normal(key, (refined.shape[0], 4))
    refined = jft.refine_healpix.refine(refined, exc, kernel)
    t = timeit(lambda: jft.refine_healpix.refine(_r, exc, kernel))
    print(
        f"{refined.shape=} time={t.time:4.2e} min={t.min:4.2e}",
        file=sys.stderr
    )

pix0s = np.stack(pixelfunc.pix2vec(1, np.arange(12), nest=nest), axis=-1)
pix0s = (
    pix0s[:, np.newaxis, :] *
    radial_chart.ind2cart(jnp.arange(n_r)[np.newaxis, :], -1)[..., np.newaxis]
).reshape(12 * n_r, 3)
cov_from_loc = _get_cov_from_loc(kernel, None)
fks_sqrt = jnp.linalg.cholesky(cov_from_loc(pix0s, pix0s))

r0 = random.normal(key, (12 * n_r, ))
refined = (fks_sqrt @ r0).reshape(12, n_r)
depth = 7
for i in range(depth):
    _r = refined
    _, key = random.split(key)
    exc = random.normal(key, (refined.shape[0], refined.shape[1] - 2, 8))
    refined = jft.refine_healpix.refine(refined, exc, kernel, radial_chart)
    t = timeit(lambda: jft.refine_healpix.refine(_r, exc, kernel, radial_chart))
    print(
        f"{refined.shape=} time={t.time:4.2e} min={t.min:4.2e}",
        file=sys.stderr
    )
