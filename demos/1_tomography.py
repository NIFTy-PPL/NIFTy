#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %% [markdown]
# # 3D Tomography Example with known start- and end-points

# %%
import jax
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from jax import numpy as jnp
from jax import random

import nifty8.re as jft

# %%
dims = (64, 64, 64)
distances = tuple(1.0 / d for d in dims)
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(offset_mean=2, offset_std=(1e-1, 3e-2))
cfm.add_fluctuations(  # Axis over which the kernle is defined
    dims,
    distances=distances,
    fluctuations=(1.0, 5e-1),
    loglogavgslope=(-5.0, 2e-1),
    flexibility=(1e0, 2e-1),
    asperity=(5e-1, 5e-2),
    prefix="ax1",
    non_parametric_kind="power",
)
correlated_field = cfm.finalize()  # forward model for a GP prior


# %%
class Forward(jft.Model):

    def __init__(self, log_density, los):
        self.log_density = log_density
        self.los = los
        # Track a method with which a random input for the model. This is not
        # strictly required but is usually handy when building deep models.
        super().__init__(init=log_density.init)

    def density(self, x):
        return jnp.exp(self.log_density(x))

    def __call__(self, x):
        return self.los(self.density(x))


key = random.PRNGKey(42)

start = (0.5, 0.5, 0.5)
# NOTE, synthetic end
n_synth_points = 100
key, sk = random.split(key)
end = random.uniform(
    sk,
    (n_synth_points, np.shape(start)[-1]),
    minval=0.05,
    maxval=0.95,
)

n_sampling_points = 256
los = jft.SamplingCartesianGridLOS(
    start,
    end,
    distances=distances,
    shape=correlated_field.target.shape,
    dtype=correlated_field.target.dtype,
    n_sampling_points=n_sampling_points,
)
forward = Forward(correlated_field, los)

# %%
key, k_f, k_n = random.split(key, 3)
synth_pos = forward.init(k_f)
synth_truth = forward(synth_pos)
noise_scl = 0.15
synth_noise = random.normal(k_n, synth_truth.shape, synth_truth.dtype)
synth_noise = synth_noise * noise_scl * synth_truth
synth_data = synth_truth + synth_noise

# %%
lh = jft.Gaussian(synth_data, noise_cov_inv=1.0 / synth_noise**2).amend(forward)

# %%
n_vi_iterations = 6
delta = 1e-4
n_samples = 4

key, k_i, k_o = random.split(key, 3)
pos_init = jft.Vector(lh.init(k_i))
samples, state = jft.optimize_kl(
    lh,
    pos_init,
    n_total_iterations=n_vi_iterations,
    n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
    key=k_o,
    draw_linear_kwargs=dict(
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None),
            maxiter=5,
        )
    ),
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M", xtol=delta, cg_kwargs=dict(name="MCG"), maxiter=35
        )
    ),
    sample_mode="linear_resample",
    odir="results_nifty_re",
    resume=False,
)

# %%
synth_density = forward.density(synth_pos)
post_density = jax.vmap(forward.density)(samples.samples)

# %%
extent = (0, 1, 0, 1)
fig, axs = plt.subplots(3, 3, dpi=500)
for i, (ax_t, ax_p, ax_ps) in enumerate(axs):
    ax_t.imshow(synth_density.sum(axis=i), extent=extent)
    # not_i = tuple(set.difference({0, 1, 2}, {i}))
    # ax_p.plot(
    #     *end[:, not_i].T,
    #     "+",
    #     markersize=0.5,
    #     color="red",
    # )
    ax_p.imshow(post_density.mean(axis=0).sum(axis=i), extent=extent)
    ax_ps.imshow(post_density.std(axis=0).sum(axis=i), extent=extent)
for title, ax in zip(("Truth", "Posterior mean", "Posterior std."), axs[0]):
    ax.set_title(title)
plt.show()

# %%
post_density_mean = post_density.mean(axis=0)

X, Y, Z = np.mgrid[
    tuple(slice(0, 1, sz * 1j) for sz in forward.log_density.target.shape)
]
n_highest_points = 10_000
q_pm = np.quantile(post_density_mean, 1 - n_highest_points / post_density_mean.size)
q_s = np.quantile(synth_density, 1 - n_highest_points / synth_density.size)
q = max(q_pm, q_s)
ss_pm = post_density_mean >= q
ss_s = synth_density >= q
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=X[ss_pm].flatten(),
            y=Y[ss_pm].flatten(),
            z=Z[ss_pm].flatten(),
            mode="markers",
            marker=dict(size=3, color="blue", opacity=0.1),
            name="Posterior Mean",
        ),
        go.Scatter3d(
            x=X[ss_s].flatten(),
            y=Y[ss_s].flatten(),
            z=Z[ss_s].flatten(),
            mode="markers",
            marker=dict(size=3, color="gray", opacity=0.1),
            name="Truth",
        ),
        # go.Scatter3d(
        #     x=end[:, 0],
        #     y=end[:, 1],
        #     z=end[:, 2],
        #     mode="markers",
        #     marker=dict(size=1),
        # )
    ]
)
axis_range = list(zip(end.min(axis=0), end.max(axis=0)))
fig.update_layout(
    showlegend=True,
    template="plotly_white",
    scene=dict(
        aspectmode="data",
        xaxis=dict(nticks=4, range=axis_range[0]),
        yaxis=dict(nticks=4, range=axis_range[1]),
        zaxis=dict(nticks=4, range=axis_range[2]),
    ),
    width=720,
    height=480,
)
fig.show()
