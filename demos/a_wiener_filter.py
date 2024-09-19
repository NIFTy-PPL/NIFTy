#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %% [markdown]
# # Demonstration of the Wiener filter and Gaussian process field model in NIFTy.re

# ## The Model

# %%
import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random

import nifty8.re as jft

jax.config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)


class FixedPowerCorrelatedField(jft.Model):
    def __init__(self, shape, *, distances=None):
        self.shape = (shape,) if isinstance(shape, int) else shape
        distances = (
            tuple((1 / s for s in self.shape)) if distances is None else distances
        )
        self.grid = jft.correlated_field.make_grid(
            dims, distances=distances, harmonic_type="fourier"
        )

        super().__init__(domain=jax.ShapeDtypeStruct(shape=dims, dtype=jnp.float64))

    def amplitude_spectrum(self):
        k = self.grid.harmonic_grid.mode_lengths
        return 0.02 / (1 + k**2)

    def correlate(self, x):
        ht = jft.correlated_field.hartley
        harmonic_dvol = 1 / self.grid.total_volume

        # Will be inlined during JIT and thus it is fine to get the amplitude
        # anew in every call
        a = self.amplitude_spectrum()
        a = a[self.grid.harmonic_grid.power_distributor]

        return harmonic_dvol * ht(a * x)

    def __call__(self, x):
        return self.correlate(x)


dims = (128, 128)
signal = FixedPowerCorrelatedField(dims)

# %% [markdown]
# ## The likelihood

# %%
signal_response = signal
noise_cov = lambda x: 0.1**2 * x
noise_cov_inv = lambda x: 0.1**-2 * x

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
# ## The Wiener filter

# %%
delta = 1e-6
key, k_w = random.split(key)
samples, info = jft.wiener_filter_posterior(
    lh,
    key=k_w,
    n_samples=20,
    draw_linear_kwargs=dict(
        cg_name="W",
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
)

# %%
post_mean, post_std = jft.mean_and_std(tuple(signal(s) for s in samples))

to_plot = [
    ("Signal", signal(pos_truth), "im"),
    ("Noise", noise_truth, "im"),
    ("Data", data, "im"),
    ("Posterior Mean", post_mean, "im"),
    ("Posterior Standard Deviation", post_std, "im"),
]
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
for ax, v in zip(axs.flat, to_plot):
    title, field, tp, *labels = v
    ax.set_title(title)
    if tp == "im":
        end = tuple(n * d for n, d in zip(signal.grid.shape, signal.grid.distances))
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
fig.savefig("wiener_filter.png", dpi=400)
plt.show()
