#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %%
from functools import partial
import sys

from jax import numpy as jnp
from jax import lax, random
from jax import jit
from jax.config import config
import matplotlib.pyplot as plt

import nifty8.re as jft

config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)


# %%
def cartesian_product(arrays, out=None):
    import numpy as np

    # Generalized N-dimensional products
    arrays = [np.asarray(x) for x in arrays]
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    if out is None:
        out = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        out[..., i] = a
    return out.reshape(-1, la)


def helper_phi_b(b, x):
    return b * x[0] * jnp.exp(b * x[1])


# %%
b = 2.

signal_response = partial(helper_phi_b, b)
nll = jft.Gaussian(0., lambda x: x / jnp.sqrt(1.)) @ signal_response

ham = jft.StandardHamiltonian(nll).jit()
ham_vg = jit(jft.mean_value_and_grad(ham))
ham_metric = jit(jft.mean_metric(ham.metric))
MetricKL = jit(
    partial(jft.MetricKL, ham),
    static_argnames=("n_samples", "mirror_samples", "linear_sampling_name"))
GeoMetricKL = partial(jft.GeoMetricKL, ham)

# %%
n_pix_sqrt = 1000
x = jnp.linspace(-4, 4, n_pix_sqrt)
y = jnp.linspace(-4, 4, n_pix_sqrt)
xx = cartesian_product((x, y))
ham_everywhere = jnp.vectorize(ham, signature="(2)->()")(xx).reshape(n_pix_sqrt, n_pix_sqrt)
plt.imshow(jnp.exp(-ham_everywhere.T), extent=(x.min(), x.max(), y.min(), y.max()), origin="lower")
plt.colorbar()
plt.title("target distribution")
plt.show()

# %%
n_mgvi_iterations = 30
n_samples = [2] * (n_mgvi_iterations-10) + [2] * 5 + [10, 10, 10, 10, 100]
n_newton_iterations = [7] * (n_mgvi_iterations-10) + [10] * 6 + 4 * [25]
absdelta = 1e-13

initial_position = jnp.array([1., 1.])
mkl_pos = 1e-2 * jft.Field(initial_position)

mgvi_positions = []

# Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    mg_samples = MetricKL(
        mkl_pos,
        n_samples[i],
        key=subkey,
        mirror_samples=True,
        linear_sampling_kwargs={"absdelta": absdelta / 10.},
    )

    print("Minimizing...", file=sys.stderr)
    opt_state = jft.minimize(
        None,
        x0=mkl_pos,
        method="newton-cg",
        options={
            "fun_and_grad": partial(ham_vg, primals_samples=mg_samples),
            "hessp": partial(ham_metric, primals_samples=mg_samples),
            "absdelta": absdelta,
            "maxiter": n_newton_iterations[i],
            "cg_kwargs": {
                "name": None
            },
            "name": "N"
        })
    mkl_pos = opt_state.x
    msg = f"Post MGVI Iteration {i}: Energy {mg_samples.at(mkl_pos).mean(ham):2.4e}"
    print(msg, file=sys.stderr)
    mgvi_positions.append(mkl_pos)

# %%
n_geovi_iterations = 15
n_samples = [1] * (n_geovi_iterations-10) + [2] * 5 + [10, 10, 10, 10, 100]
n_newton_iterations = [7] * (n_geovi_iterations-10) + [10] * 6 + [25] * 4
absdelta = 1e-10

initial_position = jnp.array([1., 1.])
gkl_pos = 1e-2 * jft.Field(initial_position)

for i in range(n_geovi_iterations):
    print(f"geoVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    geo_samples = GeoMetricKL(
        gkl_pos,
        n_samples[i],
        key=subkey,
        mirror_samples=True,
        linear_sampling_name=None,
        linear_sampling_kwargs={"absdelta": absdelta / 10.},
        non_linear_sampling_kwargs={
            "cg_kwargs": {
                "miniter": 0
            },
            "maxiter": 20
        },
    )

    print("Minimizing...", file=sys.stderr)
    opt_state = jft.minimize(
        None,
        x0=gkl_pos,
        method="newton-cg",
        options={
            "fun_and_grad": partial(ham_vg, primals_samples=geo_samples),
            "hessp": partial(ham_metric, primals_samples=geo_samples),
            "absdelta": absdelta,
            "maxiter": n_newton_iterations[i],
            "cg_kwargs": {
                "miniter": 0,
                "name": None
            },
            "name": "N"
        })
    gkl_pos = opt_state.x
    msg = f"Post geoVI Iteration {i}: Energy {geo_samples.at(gkl_pos).mean(ham):2.4e}"
    print(msg, file=sys.stderr)

# %%
n_pix_sqrt = 200
x = jnp.linspace(-4.0, 4.0, n_pix_sqrt, endpoint=True)
y = jnp.linspace(-4.0, 4.0, n_pix_sqrt, endpoint=True)
X, Y = jnp.meshgrid(x, y)
XY = jnp.array([X, Y]).T
xy = XY.reshape((XY.shape[0] * XY.shape[1], 2))
es = jnp.exp(-lax.map(ham, xy)).reshape(XY.shape[:2]).T

# %%
mkl_b_space_smpls = jnp.array([s.val for s in mg_samples.at(mkl_pos)])

fig, ax = plt.subplots()
contour = ax.contour(X, Y, es)
ax.clabel(contour, inline=True, fontsize=10)
ax.scatter(*mkl_b_space_smpls.T)
ax.plot(*mkl_pos, "rx")
plt.title("MGVI")
plt.show()

# %%
gkl_b_space_smpls = jnp.array([s.val for s in geo_samples.at(gkl_pos)])

fig, ax = plt.subplots()
contour = ax.contour(X, Y, es)
ax.clabel(contour, inline=True, fontsize=10)
ax.scatter(*gkl_b_space_smpls.T)
ax.plot(*gkl_pos, "rx")
plt.title("GeoVI")
plt.show()

# %%
initial_position = jnp.array([1., 1.])

hmc_sampler = jft.HMCChain(
    potential_energy=ham,
    inverse_mass_matrix=1.,
    position_proto=initial_position,
    step_size=0.1,
    num_steps=64,
)

chain, _ = hmc_sampler.generate_n_samples(
    42, 1e-2 * initial_position, num_samples=100, save_intermediates=True)

# %%
b_space_smpls = chain.samples
fig, ax = plt.subplots()
ax.scatter(*b_space_smpls.T)
plt.title("HMC (Metroplis-Hastings) samples")
plt.show()

# %%
initial_position = jnp.array([1., 1.])

nuts_sampler = jft.NUTSChain(
    potential_energy=ham,
    inverse_mass_matrix=0.5,
    position_proto=initial_position,
    step_size=0.4,
    max_tree_depth=10,
)

nuts_n_samples = []
ns_samples = [200, 1000, 1000000]
for n_samples in ns_samples:
    chain, _ = nuts_sampler.generate_n_samples(
        43 + n_samples, 1e-2 * initial_position, num_samples=n_samples, save_intermediates=True)
    nuts_n_samples.append(chain.samples)

# %%
b_space_smpls = chain.samples

fig, ax = plt.subplots()
contour = ax.contour(X, Y, es)
ax.clabel(contour, inline=True, fontsize=10)
ax.scatter(*b_space_smpls.T, s=2.)
plt.show()

# %%
plt.hist2d(*b_space_smpls.T, bins=[x, y], range=[[x.min(), x.max()], [y.min(), y.max()]])
plt.colorbar()
plt.show()

# %%
subplots = (3, 2)

fig_width_pt = 426    # pt (a4paper, and such)
inches_per_pt = 1 / 72.27
fig_width_in = fig_width_pt * inches_per_pt
fig_height_in = fig_width_in * 1. * (subplots[0] / subplots[1])
fig_dims = (fig_width_in, fig_height_in)

fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(
    *subplots, figsize=fig_dims, sharex=True, sharey=True)

ax1.set_title(r'$P(d=0|\xi_1, \xi_2) \cdot P(\xi_1, \xi_2)$')
xx = cartesian_product((x, y))
ham_everywhere = jnp.vectorize(ham, signature="(2)->()")(xx).reshape(n_pix_sqrt, n_pix_sqrt)
ax1.imshow(jnp.exp(-ham_everywhere.T), extent=(x.min(), x.max(), y.min(), y.max()), origin="lower")
#ax1.colorbar()

ax1.set_ylim([-4., 4.])
ax1.set_xlim([-4., 4.])
#ax1.autoscale(enable=True, axis='y', tight=True)
asp = float(jnp.diff(jnp.array(ax1.get_xlim()))[0] / jnp.diff(jnp.array(ax1.get_ylim()))[0])

smplmarkersize = .3
smplmarkercolor = 'k'

linewidths = 0.5
fontsize = 5
potlabels = False

ax2.set_title('MGVI')
mkl_b_space_smpls = jnp.array([s.val for s in mg_samples.at(mkl_pos)])
contour = ax2.contour(X, Y, es, linewidths=linewidths)
ax2.clabel(contour, inline=True, fontsize=fontsize)
ax2.scatter(*mkl_b_space_smpls.T, s=smplmarkersize, c=smplmarkercolor)
ax2.plot(*mkl_pos, "rx")
#ax2.set_aspect(asp)

ax3.set_title('geoVI')
gkl_b_space_smpls = jnp.array([s.val for s in geo_samples.at(gkl_pos)])
contour = ax3.contour(X, Y, es, linewidths=linewidths)
ax3.clabel(contour, inline=True, fontsize=fontsize)
ax3.scatter(*gkl_b_space_smpls.T, s=smplmarkersize, c=smplmarkercolor)
ax3.plot(*gkl_pos, "rx")
#ax3.set_aspect(asp)

for i in range(3):
    eval('ax' + str(i + 1)).set_ylabel(r'$\xi_2$')
ax3.set_xlabel(r'$\xi_1$')
ax6.set_xlabel(r'$\xi_1$')

for n, samples, ax in zip(ns_samples[:2], nuts_n_samples[:2], [ax4, ax5]):
    ax.set_title(f"NUTS N={n}")
    contour = ax.contour(X, Y, es, linewidths=linewidths)
    #ax.clabel(contour, inline=True, fontsize=fontsize)
    ax.scatter(*samples.T, s=smplmarkersize, c=smplmarkercolor)

h, _, _ = jnp.histogram2d(
    *nuts_n_samples[-1].T, bins=[x, y], range=[[x.min(), x.max()], [y.min(), y.max()]])
ax6.imshow(h.T, extent=(x.min(), x.max(), y.min(), y.max()), origin="lower")
ax6.set_title(f'NUTS N={ns_samples[-1]:.0E}')

fig.tight_layout()
fig.savefig("pinch.pdf", bbox_inches='tight')
print("final plot saved as pinch.pdf")
