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

seed = 100
key = random.PRNGKey(seed)

dims = (128, 128)

cf_zm = dict(offset_mean=0.0, offset_std=(1e-3, 1e-4))
cf_fl = dict(
    fluctuations=(1e-1, 1e-16),
    loglogavgslope=(-4.0, 1e-16),
    flexibility=(10, 1e-16),
    asperity=None,
    # flexibility=None,
    # asperity=None,
)

# cf_fl = dict(
#     fluctuations=(1e-1, 5e-3),
#     loglogavgslope=(-1.0, 1e-2),
#     flexibility=(1e0, 5e-1),
#     asperity=(5e-1, 5e-2),
# )

cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(
    dims, distances=1.0 / dims[0], **cf_fl, prefix="ax1", non_parametric_kind="power"
)
correlated_field = cfm.finalize()

scaling = jft.LogNormalPrior(3.0, 1.0, name="scaling", shape=(1,))


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
# ## The inference

# %%
n_vi_iterations = 10
delta = 1e-4
n_samples = 4

key, k_i, k_o = random.split(key, 3)
# NOTE, changing the number of samples always triggers a resampling even if
# `resamples=False`, as more samples have to be drawn that did not exist before.
samples, state = jft.optimize_kl(
    lh,
    jft.Vector(lh.init(k_i)),
    n_total_iterations=n_vi_iterations,
    n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
    # Source for the stochasticity for sampling
    key=k_o,
    # Names of parameters that should not be sampled but still optimized
    # can be specified as point_estimates (effectively we are doing MAP for
    # these degrees of freedom).
    # point_estimates=("cfax1flexibility", "cfax1asperity"),
    # Arguments for the conjugate gradient method used to drawing samples
    draw_linear_kwargs=dict(
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
    # Arguements for the minimizer in the nonlinear updating of the samples
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None),
            maxiter=100,
        )
    ),
    # Arguments for the minimizer of the KL-divergence cost potential
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=35
        )
    ),
    sample_mode="nonlinear_resample",
    odir="results_intro",
    resume=False,
    implicit_samples="standard"
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
        "Amplitude spectrum mean",
        (
            grid.harmonic_grid.mode_lengths[1:],
            cfm.amplitude(pos_truth)[1:],
            post_a_mean,
        ),
        "loglog",
    ),
    (
        "Amplitude spectrum samples",
        (grid.harmonic_grid.mode_lengths[1:],) + tuple(cfm.amplitude(s)[1:] for s in samples),
        "loglog",
    ),
]


im_fields = [field for title, field, tp, *_ in to_plot if tp == "im" and title != "Noise"]
vmin = min(f.min() for f in im_fields)
vmax = max(f.max() for f in im_fields)

fig, axs = plt.subplots(2, 3, figsize=(16, 9))
for ax, v in zip(axs.flat, to_plot):
    title, field, tp, *labels = v
    ax.set_title(title)
    if tp == "im":
        end = tuple(n * d for n, d in zip(grid.shape, grid.distances))
        if title == "Noise":
            im = ax.imshow(field.T, cmap="inferno", extent=(0.0, end[0], 0.0, end[1]))
        else:
            im = ax.imshow(
                field.T,
                cmap="inferno",
                extent=(0.0, end[0], 0.0, end[1]),
                vmin=vmin,
                vmax=vmax,
            )
        plt.colorbar(im, ax=ax, orientation="horizontal")
    else:
        ax_plot = ax.loglog if tp == "loglog" else ax.plot
        x = field[0]
        if title == "Amplitude spectrum mean":
            ax_plot(x, field[1], alpha=0.7, label="Truth")
            ax_plot(x, field[2], alpha=0.7, label="Posterior mean")
            ax.legend()
        else:
            for f in field[1:]:
                ax_plot(x, f, alpha=0.7)

for ax in axs.flat[len(to_plot):]:
    ax.set_axis_off()
fig.tight_layout()

from datetime import datetime
timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
filename = f"scatter_{timestamp}.png"
plt.savefig(f"demos/Elias/0_intro_data/{filename}", dpi=600, bbox_inches="tight")
plt.show()
