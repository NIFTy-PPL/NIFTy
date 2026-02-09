#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %% [markdown]
# # Demonstration of the non-parametric correlated field model in NIFTy.re

# ## The Model

# %%
import jax
import matplotlib.pyplot as plt
import nifty.re as jft
from jax import numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)

dims = (32, 32)
# dims = (64, 64)

cf_zm = dict(offset_mean=-4.0, offset_std=(2, 0.1))
cf_fl = dict(
    fluctuations=(1, 0.5),
    loglogavgslope=(-3.5, 0.1),
    flexibility=(2, 1),
    asperity=(0.5, 0.1),
)
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(
    dims, distances=1.0 / dims[0], **cf_fl, prefix="ax1", non_parametric_kind="power"
)
correlated_field = cfm.finalize()

scaling = jft.LogNormalPrior(2.0, 1.0, name="scaling", shape=(1,))


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
# ## The likelihood

# %%
signal_response = signal
noise_std = 0.03
noise_cov = lambda x: noise_std**2 * x
noise_cov_inv = lambda x: noise_std**-2 * x

# Create synthetic data
key, subkey = random.split(key)
pos_truth = jft.random_like(subkey, signal_response.domain)
signal_response_truth = signal_response(pos_truth)
key, subkey = random.split(key)
noise_truth = (
    (noise_cov(jft.ones_like(signal_response.target))) ** 0.5
) * jft.random_like(key, signal_response.target)
data = signal_response_truth + noise_truth

lh = jft.Gaussian(data, noise_cov_inv).amend(signal_response)

# %% [markdown]
# ## The inference

# %%
key, key_init, key_nuts = random.split(key, 3)
pos_init = lh.init(key_init)

samples, state = jft.blackjax_nuts(
    lh,
    pos_init,
    key_nuts,
    n_warmup_steps=1_000,
    n_samples=1_000,
)


# %%
namps = cfm.get_normalized_amplitudes()
post_sr_mean = jft.mean(tuple(signal(s) for s in samples))
post_a_mean = jft.mean(tuple(cfm.amplitude(s)[1:] for s in samples))
grid = correlated_field.target_grids[0]
to_plot = [
    ("Signal", signal(pos_truth), "im"),
    ("Noise", noise_truth, "im"),
    ("Data", data, "im"),
    ("Reconstruction", post_sr_mean, "im"),
    (
        "Amplitude spectrum",
        (
            grid.harmonic_grid.mode_lengths[1:],
            cfm.amplitude(pos_truth)[1:],
            post_a_mean,
        ),
        "loglog",
    ),
]
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
for ax, v in zip(axs.flat, to_plot):
    title, field, tp, *labels = v
    ax.set_title(title)
    if tp == "im":
        end = tuple(n * d for n, d in zip(grid.shape, grid.distances))
        im = ax.imshow(field.T, cmap="inferno", extent=(0.0, end[0], 0.0, end[1]))
        plt.colorbar(im, ax=ax, orientation="horizontal")
    else:
        ax_plot = ax.loglog if tp == "loglog" else ax.plot
        x = field[0]
        for f in field[1:]:
            ax_plot(x, f, alpha=0.7)
for ax in axs.flat[len(to_plot) :]:
    ax.set_axis_off()
fig.tight_layout()
fig.savefig("results_intro_full_reconstruction.png", dpi=400)
plt.show()
