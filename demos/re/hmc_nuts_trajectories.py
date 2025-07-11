#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %%
#
# WARNING: This code does not behave deterministically. It works fine when
# executing cell by cell using vscodes notebook functionality but when running
# from the command line with either python3 or ipython3 the following happens:
# This is probably due to an issue with host_callback.
# Concretely it just stops adding points to the debug list after some random
# number of leapfrog steps.
#

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt

import nifty8.re as jft

# %%
jft.hmc._DEBUG_FLAG = True

# %%
cov = jnp.array([10., 1.])

potential_energy = lambda q: jnp.sum(0.5 * q**2 / cov)

initial_position = jnp.array([1., 1.])

sampler = jft.NUTSChain(
    potential_energy=potential_energy,
    inverse_mass_matrix=1.,
    position_proto=initial_position,
    step_size=0.12,
    max_tree_depth=10,
)

# %%
jft.hmc._DEBUG_STORE = []
jft.hmc._DEBUG_TREE_END_IDXS = []
jft.hmc._DEBUG_SUBTREE_END_IDXS = []

chain, _ = sampler.generate_n_samples(
    48, initial_position, num_samples=5, save_intermediates=True
)

plt.hist(chain.depths)
plt.show()

# %%
debug_pos = jnp.array([qp.position for qp in jft.hmc._DEBUG_STORE])
print(len(debug_pos))

# %%
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

ax = plt.gca()
ellipse = matplotlib.patches.Ellipse(
    xy=(0, 0),
    width=jnp.sqrt(cov[0]),
    height=jnp.sqrt(cov[1]),
    edgecolor='k',
    fc='None',
    lw=1
)
ax.add_patch(ellipse)

color_idx = 0
start_and_end_idxs = zip(
    [
        0,
    ] + jft.hmc._DEBUG_SUBTREE_END_IDXS[:-1], jft.hmc._DEBUG_SUBTREE_END_IDXS
)
for start_idx, end_idx in start_and_end_idxs:
    slice = debug_pos[start_idx:end_idx]
    ax.plot(
        slice[:, 0],
        slice[:, 1],
        '-o',
        markersize=1,
        linewidth=0.5,
        color=colors[color_idx % len(colors)]
    )
    if end_idx in jft.hmc._DEBUG_TREE_END_IDXS:
        color_idx = (color_idx + 1) % len(colors)

ax.scatter(
    chain.samples[:, 0],
    chain.samples[:, 1],
    marker='x',
    color='k',
    label='samples'
)
ax.scatter(initial_position[0], initial_position[1], label='starting position')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

fig_width_pt = 426  # pt (a4paper, and such)
# fig_width_pt = 360 # pt
inches_per_pt = 1 / 72.27
fig_width_in = 0.9 * fig_width_pt * inches_per_pt
fig_height_in = fig_width_in * 0.618
fig_dims = (fig_width_in, fig_height_in)

plt.tight_layout()
plt.show()
plt.savefig("trajectories.pdf", bbox_inches='tight')
