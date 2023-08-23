#!/usr/bin/env python3
# Copyright(C) 2013-2023 Gordian Edenhofer
# SPDX-License-Identifier: BSD-2-Clause

# %%
import sys
from functools import partial

import jax
import matplotlib.pyplot as plt
import nifty8.re as jft
import numpy as np
from jax import numpy as jnp
from jax import random
from jax.config import config

config.update("jax_enable_x64", True)


def get_power_spectrum(cutoff, scale, negloglogslope, high_res_cutoff=1e-3):
    if all(x is None for x in (cutoff, scale, negloglogslope)):
        cutoff, scale, negloglogslope = jnp.exp(
            jnp.array([-6.47776258, 16.43005412, 1.16049258])
        )

    def power_spectrum(k):
        return scale / (1 + (k / cutoff)**
                        negloglogslope) * jnp.exp(-high_res_cutoff * k)

    return power_spectrum


def get_kernel(*args, **kwargs):
    import numpy as np
    from hankel import SymmetricFourierTransform

    ps = get_power_spectrum(*args, **kwargs)

    def kernel_via_hankel(x, dim=3):
        # Hankel's SymmetricFourierTransform can't handle the zero-mode, thus
        # simply add it afterwards.
        ft = SymmetricFourierTransform(ndim=dim, N=30000, h=5e-9)
        Fk = ft.transform(lambda x: np.array(ps(x)), np.array(x), inverse=False)
        return Fk[0] + 1.431e9 / (540 * 740**2)  # add zero mode

    xp = jnp.logspace(-6, 11, 2248, base=jnp.e)
    yp = kernel_via_hankel(xp)
    zm = kernel_via_hankel(0)
    xp = jnp.append(0, xp)
    yp = jnp.log(jnp.append(zm, yp))

    def interp(x):
        return jnp.exp(jnp.interp(x, xp, yp))

    return interp


# %%
# The log of these parameters has been fit on previous reconstructions
kernel_parameters = {
    "cutoff": 1.5372463000446132e-3,
    "scale": 1.3660979564940365e+7,
    "negloglogslope": 3.191504960507704,
    "high_res_cutoff": 1.e-3,
}

cloud = ((170, 180), (-10, -20))  # Taurus
min_shape = (32, 32, 32)

kernel = get_kernel(**kernel_parameters)
r = 300
extent_pos_deg = tuple(np.abs(el[1] - el[0]) for el in cloud)
extent = tuple(np.tan(np.radians(d)) * r for d in extent_pos_deg)
extent += (extent[0], )
distances = tuple(e / s for e, s in zip(extent, min_shape))

cc = jft.CoordinateChart(
    min_shape=min_shape,
    distances=distances,
    irregular_axes=(),
)
rf = jft.RefinementField(cc)


def dust_density_apply(exc, **kw):
    return jnp.exp(rf(exc, **kw))


def projected_dust_density_apply(exc, **kw):
    return dust_density_apply(exc, **kw).sum(axis=-1)


# Make the refinement fast by leaving the kernel fixed
rfm = rf.matrices(kernel)
signal = jft.Model(
    partial(projected_dust_density_apply, kernel=rfm),
    domain=rf.domain,
    init=rf.init
)

# %%
# Generate the Poisson data
key = random.PRNGKey(0)
key, key_signal_truth, key_noise_truth = random.split(key, 3)
signal_truth = signal(signal.init(key_signal_truth))
noise = random.normal(key_noise_truth, signal_truth.shape)
noise *= np.sqrt(signal_truth)
data = signal_truth + noise

# Define the noise solely via the data assuming only Poissonian noise statistics
# and no knowledge about the actual data
noise_bound = np.quantile(data[data > 0], 1e-2)
noise_cov_val = data.clip(noise_bound, None)
noise_cov = lambda x: noise_cov_val * x
noise_cov_inv = lambda x: x / noise_cov_val
nll = jft.Gaussian(data, noise_cov_inv) @ signal


# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
im = axs[0].imshow(data, vmin=0., vmax=data.max())
axs[1].imshow(signal_truth, vmin=0., vmax=data.max())
fig.colorbar(im, ax=axs.ravel())
plt.show()

def plot(pos, samples):
    fig, axs = plt.subplots(1, 3, figsize=(8, 3), dpi=500)
    im = axs[0].imshow(data, vmin=0., vmax=data.max())
    axs[1].imshow(signal_truth, vmin=0., vmax=data.max())
    axs[2].imshow(
        jax.vmap(signal)(samples.at(pos).samples).mean(axis=0),
        vmin=0.,
        vmax=data.max()
    )
    fig.colorbar(im, ax=axs.ravel())
    plt.show()


# %%
key, subkey = random.split(key)
pos_init = jft.random_like(subkey, signal.domain)
pos = 1e-2 * jft.Vector(pos_init.copy())

n_iter = 30
absdelta = 1e-4 * jnp.prod(jnp.array(min_shape))
key, subkey = random.split(key)

res, samples = jft.optimize_kl(nll, pos, 10, 2, subkey,
    sampling_method='altmetric',
    sampling_kwargs={
        'name':'Sampling',
        'xtol': 0.001,
        'maxiter':n_iter,
        'cg_kwargs':{'name':None}},
    sampling_cg_kwargs={'maxiter':50,
                        'name':'Sampling linear'},
    minimization_kwargs={
        'name':'minimize',
        "absdelta": absdelta,
        'maxiter':n_iter},
    out_dir='altres',
    resume=False,
    verbosity=0
)

# Split key once more to have the same random numbers as `optimize_kl`
_, subkey = random.split(subkey)
ovi = jft.OptimizeVI(nll, 10, subkey, 2, 
                 sampling_method='altmetric',
                 sampling_kwargs={
                     'name':'Sampling',
                     'xtol': 0.001,
                     'maxiter':n_iter,
                     'cg_kwargs':{'name':None}},
                 sampling_cg_kwargs={'maxiter':50,
                                     'name':'Sampling linear'},
                 minimization_kwargs={
                     'name':'minimize',
                     "absdelta": absdelta,
                     'maxiter':n_iter,
                     'cg_kwargs':{'name':None}})
res2, state = ovi.run(pos)

plot(res, samples)
plot(res2, state.samples)
