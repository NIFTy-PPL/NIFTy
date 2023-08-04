#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %%
import sys
from functools import partial

import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random
from jax.config import config

import nifty8.re as jft

config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)

dims = (128, 128)

n_mgvi_iterations = 3
n_samples = 4
n_newton_iterations = 10
absdelta = 1e-4 * jnp.prod(jnp.array(dims))

cf_zm = {"offset_mean": 0., "offset_std": (1e-3, 1e-4)}
cf_fl = {
    "fluctuations": (1e-1, 5e-3),
    "loglogavgslope": (-1., 1e-2),
    "flexibility": (1e+0, 5e-1),
    "asperity": (5e-1, 5e-2),
    "harmonic_domain_type": "Fourier"
}
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(
    dims,
    distances=1. / dims[0],
    **cf_fl,
    prefix="ax1",
    non_parametric_kind="power"
)
correlated_field = cfm.finalize()

signal = jft.Model(
    lambda x: jnp.exp(correlated_field(x)),
    domain=correlated_field.domain,
    init=correlated_field.init
)

# %% [markdown]
# ## NIFTy to NIFTY.re

# The equivalent model in numpy-based NIFTy reads

# ```python
# import nifty8 as ift
#
# position_space = ift.RGSpace(dims, distances=1. / dims[0])
# cf_fl_nft = {
#     k: v
#     for k, v in cf_fl.items() if k not in ("harmonic_domain_type", )
# }
# cfm_nft = ift.CorrelatedFieldMaker("cf")
# cfm_nft.add_fluctuations(position_space, **cf_fl_nft, prefix="ax1")
# cfm_nft.set_amplitude_total_offset(**cf_zm)
# correlated_field_nft = cfm_nft.finalize()
# signal_nft = correlated_field_nft.exp()
#```

# For convience, NIFTy implements a method to translate numpy-based NIFTy
# operators to NIFTy.re. One can access the equivalent expression in JAX for a
# NIFTy model via the `.jax_expr` property of an operator. In addition, NIFTy
# features a method to additionally preserve the domain and target:
# `ift.nifty2jax.convert` translate NIFTy operators to `jft.Model`. NIFTy.re
# models feature `.domain` and `.target` properties but instead of yielding
# domains, they return [JAX PyTrees](TODO:cite PyTree docu) of shape-and-dtype
# objects.

#```python
# # Convenience method to get JAX expression as NIFTy.re model which tracks
# # domain and target
# signal_nft: jft.Model = ift.nifty2jax.convert(signal_nft, float)
# ```

# Both expressions are identical up to floating point precision
# ```python
# import numpy as np
#
# t = signal.init(random.PRNGKey(42))
# np.testing.assert_allclose(signal(t), signal_nft(t), atol=1e-13, rtol=1e-13)
# ```

# Note, caution is advised when translating NIFTy models working on complex
# numbers. Numyp-based NIFTy models are not dtype aware and thus require more
# care when translating them to NIFTy.re/JAX which requires known dtypes.

# %% [markdown]
# ## Notes on Refinement Field

# The above could cust as well be a refinement field e.g. on a HEALPix sphere
# with logarithmically spaced radial voxels. All of NIFTy.re is agnostic to the
# specifics of the forward model. The sampling and minimization always works the
# same.

# %%
# def matern_kernel(distance, scale=1., cutoff=1., dof=1.5):
#     if dof == 0.5:
#         cov = scale**2 * jnp.exp(-distance / cutoff)
#     elif dof == 1.5:
#         reg_dist = jnp.sqrt(3) * distance / cutoff
#         cov = scale**2 * (1 + reg_dist) * jnp.exp(-reg_dist)
#     elif dof == 2.5:
#         reg_dist = jnp.sqrt(5) * distance / cutoff
#         cov = scale**2 * (1 + reg_dist + reg_dist**2 / 3) * jnp.exp(-reg_dist)
#     else:
#         raise NotImplementedError()
#     # NOTE, this is not safe for differentiating because `cov` still may
#     # contain NaNs
#     return jnp.where(distance < 1e-8 * cutoff, scale**2, cov)

# def rg2cart(x, idx0, scl):
#     """Transforms regular, points from a Euclidean space to irregular points in
#     an cartesian coordinate system in 1D."""
#     return jnp.exp(scl * x[0] + idx0)[jnp.newaxis, ...]

# def cart2rg(x, idx0, scl):
#     """Inverse of `rg2cart`."""
#     return ((jnp.log(x[0]) - idx0) / scl)[jnp.newaxis, ...]

# cc = jft.HEALPixChart(
#     min_shape=(12 * 32**2, 4),  # 32 (Nside) times (at least) 4 radial bins
#     nonhp_rg2cart=partial(rg2cart, idx0=-0.27, scl=1.1),  # radial spacing
#     nonhp_cart2rg=partial(cart2rg, idx0=-0.27, scl=1.1),
# )
# rf = jft.RefinementHPField(cc)
# # Make the refinement fast by leaving the kernel fixed
# rfm = rf.matrices(matern_kernel)
# correlated_field = jft.Model(
#     partial(rf, kernel=rfm), domain=rf.domain, init=rf.init
# )

# %%
signal_response = signal
noise_cov = lambda x: 0.1**2 * x
noise_cov_inv = lambda x: 0.1**-2 * x

# Create synthetic data
key, subkey = random.split(key)
pos_truth = jft.random_like(subkey, signal_response.domain)
signal_response_truth = signal_response(pos_truth)
key, subkey = random.split(key)
noise_truth = jnp.sqrt(
    noise_cov(jnp.ones(signal_response.target.shape))
) * random.normal(shape=signal_response.target.shape, key=key)
data = signal_response_truth + noise_truth

nll = jft.Gaussian(data, noise_cov_inv) @ signal_response
# NOTE, keep in mind that sampling with geoVI can currently not be compiled.
# Jit-compile the next best thing: the likelihood, its metric and its
# left-sqrt-metric. This is redundant for MGVI.
nll = nll.jit()

key, subkey = random.split(key)
pos_init = jft.random_like(subkey, correlated_field.domain)
linear_sampling_kwarks = {"absdelta": absdelta / 10., "maxiter": 100}
non_linear_sampling_kwarks = None # For MGVI
#non_linear_sampling_kwarks = {"absdelta": absdelta / 10., "maxiter": 10} # For geoVI
newton_cg_kwarks = {"absdelta": absdelta, "maxiter": n_newton_iterations}
n_samples = lambda i: 2 if i<2 else 4
pos, samples, key = jft.optimize_kl(nll,
                             pos_init,
                             n_mgvi_iterations,
                             n_samples,
                             newton_cg_kwarks,
                             linear_sampling_kwarks,
                             non_linear_sampling_kwarks,
                             key)
# %%
namps = cfm.get_normalized_amplitudes()
post_sr_mean = jft.mean(tuple(signal_response(s) for s in samples.at(pos)))
post_a_mean = jft.mean(tuple(cfm.amplitude(s)[1:] for s in samples.at(pos)))
to_plot = [
    ("Signal", signal_response_truth, "im"),
    ("Noise", noise_truth, "im"),
    ("Data", data, "im"),
    ("Reconstruction", post_sr_mean, "im"),
    ("Ax1", (cfm.amplitude(pos_truth)[1:], post_a_mean), "loglog"),
]
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
for ax, (title, field, tp) in zip(axs.flat, to_plot):
    ax.set_title(title)
    if tp == "im":
        im = ax.imshow(field, cmap="inferno")
        plt.colorbar(im, ax=ax, orientation="horizontal")
    else:
        ax_plot = ax.loglog if tp == "loglog" else ax.plot
        field = field if isinstance(field, (tuple, list)) else (field, )
        for f in field:
            ax_plot(f, alpha=0.7)
fig.tight_layout()
fig.savefig("cf_w_unknown_spectrum.png", dpi=400)
plt.close()
