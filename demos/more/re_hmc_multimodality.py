#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %%
from functools import partial

import jax.numpy as jnp
import matplotlib.pyplot as plt

import nifty8.re as jft


def loggaussian(x, mu, sigma):
    return -0.5 * (x - mu)**2 / sigma


def sum_of_gaussians(x, separation, sigma1, sigma2):
    return -jnp.logaddexp(
        loggaussian(x, 0, sigma1), loggaussian(x, separation, sigma2)
    )


ham = partial(sum_of_gaussians, separation=10., sigma1=1., sigma2=1.)

N = 100000
SEED = 43
EPS = 0.3

subplots = (2, 2)
fig_width_pt = 426  # pt (a4paper, and such)
# fig_width_pt = 360 # pt
inches_per_pt = 1 / 72.27
fig_width_in = 0.9 * fig_width_pt * inches_per_pt
fig_height_in = fig_width_in * 0.618 * (subplots[0] / subplots[1])
fig_dims = (fig_width_in, fig_height_in * 1.5)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    subplots[0],
    subplots[1],
    sharex='col',
    figsize=fig_dims,
    gridspec_kw={'width_ratios': [1, 2]}
)

# %%
nuts_sampler = jft.NUTSChain(
    potential_energy=ham,
    inverse_mass_matrix=5.,
    position_proto=jnp.array(0.),
    step_size=EPS,
    max_tree_depth=15,
    max_energy_difference=1000.,
)

chain, _ = nuts_sampler.generate_n_samples(
    SEED, jnp.array(3.), num_samples=N, save_intermediates=True
)
print(f"small mass matrix acceptance: {chain.acceptance}")

ax1.hist(chain.samples, bins=30, density=True)
ax2.plot(chain.samples, linewidth=0.5)

ax1.set_title(rf'$m={1. / nuts_sampler.inverse_mass_matrix:1.2f}$')
ax2.set_title(rf'$m={1. / nuts_sampler.inverse_mass_matrix:1.2f}$')

# %%
nuts_sampler = jft.NUTSChain(
    potential_energy=ham,
    inverse_mass_matrix=50.,
    position_proto=jnp.array(0.),
    step_size=EPS,
    max_tree_depth=15,
    max_energy_difference=1000.,
)

chain, _ = nuts_sampler.generate_n_samples(
    SEED, jnp.array(3.), num_samples=N, save_intermediates=True
)
print(f"large mass matrix acceptance: {chain.acceptance}")

ax3.hist(chain.samples, bins=30, density=True)
ax4.plot(chain.samples, linewidth=0.5)

ax3.set_title(rf'$m={1. / nuts_sampler.inverse_mass_matrix:1.2f}$')
ax4.set_title(rf'$m={1. / nuts_sampler.inverse_mass_matrix:1.2f}$')

# %%
xs = jnp.linspace(-10, 20, num=500)
Z = jnp.trapz(jnp.exp(-ham(xs)), xs)
ax1.plot(xs, jnp.exp(-ham(xs)) / Z, linewidth=0.5, c='r')
ax3.plot(xs, jnp.exp(-ham(xs)) / Z, linewidth=0.5, c='r')

ax1.set_ylabel('frequency')
ax2.set_ylabel('position')
ax3.set_xlabel('position')
ax3.set_ylabel('frequency')
ax4.set_xlabel('time')
ax4.set_ylabel('position')

#fig.suptitle("sum of two Gaussians, with different choices of mass matrix")

fig.tight_layout()
fig.savefig("multimodal.pdf", bbox_inches='tight')
print("final figure saved as multimodal.pdf")
