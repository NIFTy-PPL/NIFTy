#!/usr/bin/env python3
# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# This demo illustrates how to use multiple GPUs with NIFTy. In particular,
# it demonstrates how to distribute the samples of `optimize_kl` across
# multiple devices. The comments in this script focus on the steps required
# for running a multi-GPU reconstruction. For a general introduction and
# conceptual explanations, refer to the `0_intro.py` demo, which provides a
# single-device version.


# NOTE: Development only — remove when running on actual multi-GPU hardware.
# This environment variable forces JAX to create 8 virtual CPU devices,
# enabling testing of NIFTy.re's multi-device execution on a single machine
# without requiring physical GPUs. Make sure to remove this setting when running
# on real multi-GPU systems.
import os

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=4"  # Use 4 CPU devices
)


import jax
import matplotlib.pyplot as plt
import nifty.re as jft
from jax import numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)

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

scaling = jft.LogNormalPrior(3.0, 1.0, name="scaling", shape=(1,))


class Signal(jft.Model):
    def __init__(self, correlated_field, scaling):
        self.cf = correlated_field
        self.scaling = scaling
        super().__init__(init=self.cf.init | self.scaling.init)

    def __call__(self, x):
        return self.scaling(x) * jnp.exp(self.cf(x))


signal = Signal(correlated_field, scaling)


signal_response = signal
noise_cov = lambda x: 0.1**2 * x
noise_cov_inv = lambda x: 0.1**-2 * x

key, subkey = random.split(key)
pos_truth = jft.random_like(subkey, signal_response.domain)
signal_response_truth = signal_response(pos_truth)
key, subkey = random.split(key)
noise_truth = (
    (noise_cov(jft.ones_like(signal_response.target))) ** 0.5
) * jft.random_like(key, signal_response.target)
data = signal_response_truth + noise_truth

lh = jft.Gaussian(data, noise_cov_inv).amend(signal_response)

n_vi_iterations = 6
delta = 1e-4


key, k_i, k_o = random.split(key, 3)
samples, state = jft.optimize_kl(
    lh,
    jft.Vector(lh.init(k_i)),
    n_total_iterations=n_vi_iterations,
    n_samples=4,
    key=k_o,
    # Use the static conjugate gradient solver for the linear sampling step, as
    # it must be JIT-compilable for multi-GPU execution.
    draw_linear_kwargs=dict(
        cg=jft.conjugate_gradient.static_cg,
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
    # Use static newton conjugate gradient for the nonlinear update step, as it
    # must to be jit compilable for multi-gpu execution.
    nonlinearly_update_kwargs=dict(
        minimize=jft.optimize._static_newton_cg,
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None),
            maxiter=5,
        ),
    ),
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=35
        )
    ),
    sample_mode="nonlinear_resample",
    odir="results_intro_multi-gpu",
    resume=False,
    # To map the sampling over devices JAX needs to trace the sampling step.
    # Therefore you need to use `smap` or `vmap` as a residual map function.
    residual_map="vmap",
    devices=jax.devices(),
)

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
