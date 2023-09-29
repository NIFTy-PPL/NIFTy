#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %%
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random
from jax.config import config

import nifty8.re as jft

config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)

dims = (128, 128)

n_vi_iterations = 6
n_newton_iterations = 10
delta = 1e-4
absdelta = delta * jnp.prod(jnp.array(dims))

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

scaling = jft.LogNormalPrior(3., 1., name="scaling", shape=(1, ))


class Signal(jft.Model):
    def __init__(self, correlated_field, scaling):
        self.cf = correlated_field
        self.scaling = scaling
        # Init methods of the Correlated Field model and any prior model in
        # NIFTy.re are aware that their input is standard normal a priori.
        # The `domain` of a model does not know this. Thus, tracking the `init`
        # methods should be preferred over tracking the `domain`.
        super().__init__(init=self.cf.init | self.scaling.init)

    def __call__(self, x):
        # NOTE, think of `Model` as being just a plain function that takes some
        # input and performs all the necessary computation for your model.
        # Note, `scaling` here is completely degenarate with `offset_std` in the
        # likelihood but the priors for them are very different.
        return self.scaling(x) * jnp.exp(self.cf(x))


signal = Signal(correlated_field, scaling)

# %% [markdown]
# ## NIFTy to NIFTY.re

# The equivalent model for the correlated field in numpy-based NIFTy reads

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
# correlated_field_nft: jft.Model = ift.nifty2jax.convert(
#     correlated_field_nft, float
# )
# ```

# Both expressions are identical up to floating point precision
# ```python
# import numpy as np
#
# t = correlated_field_nft.init(random.PRNGKey(42))
# np.testing.assert_allclose(
#     correlated_field(t), correlated_field_nft(t), atol=1e-13, rtol=1e-13
# )
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
noise_truth = ((noise_cov(jft.ones_like(signal_response.target)))**0.5
) * jft.random_like(key, signal_response.target)
data = signal_response_truth + noise_truth

nll = jft.Gaussian(data, noise_cov_inv) @ signal_response

key, subkey = random.split(key)
pos_init = jft.random_like(subkey, signal_response.domain)
pos_init = jft.Vector(pos_init)
linear_sampling_kwarks = {"absdelta": absdelta / 10., "maxiter": 100}
sampling_kwargs = {"xtol": delta, "maxiter": 10}
minimization_kwarks = {"absdelta": absdelta, "maxiter": n_newton_iterations}
# NOTE, changing the number of samples always triggers a resampling even if
# `resamples=False`, as more samples have to be drawn that did not exist before.
n_samples = 4
pos, samples = jft.optimize_kl(nll, pos_init,
                               n_vi_iterations,
                               n_samples,
                               key,
                               minimizer='newtoncg',
                               minimization_kwargs=minimization_kwarks,
                               sampling_method='altmetric',
                               # 'linear' for MGVI, 'geometric' for geoVI
                               sampling_minimizer='newtoncg',
                               sampling_kwargs=sampling_kwargs,
                               sampling_cg_kwargs=linear_sampling_kwarks,
                               resample=lambda ii: True if ii<2 else False,
                               out_dir="results_jifty",
                               verbosity=0)
# %%
namps = cfm.get_normalized_amplitudes()
post_sr_mean = jft.mean(tuple(signal(s) for s in samples.at(pos)))
post_a_mean = jft.mean(tuple(cfm.amplitude(s)[1:] for s in samples.at(pos)))
to_plot = [
    ("Signal", signal(pos_truth), "im"),
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

# %%
