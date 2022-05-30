#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %%
from functools import partial
import sys

from jax import numpy as jnp
from jax import lax, random
from jax import jit, value_and_grad
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


def banana_helper_phi_b(b, x):
    return jnp.array([x[0], x[1] + b * x[0]**2 - 100*b])


# %%
b = 0.1

SCALE = 10.

signal_response = lambda s: banana_helper_phi_b(b, SCALE * s)
nll = jft.Gaussian(jnp.zeros(2), lambda x: x / jnp.array([100., 1.])) @ signal_response
nll = nll.jit()
nll_vg = jit(value_and_grad(nll))

ham = jft.StandardHamiltonian(nll)
ham = ham.jit()
ham_vg = jit(jft.mean_value_and_grad(ham))
ham_metric = jit(jft.mean_metric(ham.metric))

MetricKL = jit(
    partial(jft.MetricKL, ham),
    static_argnames=("n_samples", "mirror_samples", "linear_sampling_name"))
GeoMetricKL = partial(jft.GeoMetricKL, ham)

# # %%
# # TODO: Stabilize inversion
# gkl_position = jnp.array([1.15995025, -0.35110244])
# special_key = jnp.array([3269562362, 460782344], dtype=jnp.uint32)
# err = jft.geometrically_sample_standard_hamiltonian(
#     key=special_key,
#     hamiltonian=ham,
#     primals=gkl_position,
#     mirror_linear_sample=False,
#     linear_sampling_name="SCG",
#     linear_sampling_kwargs={"miniter": -1},
#     non_linear_sampling_name="S",
#     non_linear_sampling_kwargs={
#         "cg_kwargs": {
#             "miniter": -1
#         },
#         "maxiter": 20,
#     }
# )

# %%  # MGVI
n_mgvi_iterations = 30
n_samples = [1] * (n_mgvi_iterations-2) + [2] + [100]
n_newton_iterations = [7] * (n_mgvi_iterations-10) + [10] * 6 + 4 * [25]
absdelta = 1e-10

initial_position = jnp.array([1., 1.])
mkl_pos = 1e-2 * initial_position

# Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    mg_samples = MetricKL(
        mkl_pos,
        n_samples=n_samples[i],
        key=subkey,
        mirror_samples=True,
        linear_sampling_kwargs={"miniter": 0})

    print("Minimizing...", file=sys.stderr)
    opt_state = jft.minimize(
        None,
        x0=mkl_pos,
        method="newton-cg",
        options={
            "fun_and_grad": partial(ham_vg, primals_samples=mg_samples),
            "hessp": partial(ham_metric, primals_samples=mg_samples),
            "energy_reduction_factor": None,
            "absdelta": absdelta,
            "maxiter": n_newton_iterations[i],
            "cg_kwargs": {
                "miniter": 0,
                "name": None
            },
            "name": "N"
        })
    mkl_pos = opt_state.x
    print((f"Post MGVI Iteration {i}: Energy {mg_samples.at(mkl_pos).mean(ham):2.4e}"
           f"; #NaNs {jnp.isnan(mkl_pos).sum()}"),
          file=sys.stderr)

# %%  # geoVI
n_geovi_iterations = 15
n_samples = [1] * (n_geovi_iterations-2) + [2] + [100]
n_newton_iterations = [7] * (n_geovi_iterations-10) + [10] * 6 + [25] * 4
absdelta = 1e-10

initial_position = jnp.array([1., 1.])
gkl_pos = 1e-2 * initial_position

for i in range(n_geovi_iterations):
    print(f"GeoVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    geo_samples = GeoMetricKL(
        gkl_pos,
        n_samples[i],
        key=subkey,
        mirror_samples=True,
        linear_sampling_name=None,
        linear_sampling_kwargs={"miniter": 0},
        non_linear_sampling_name=None,
        non_linear_sampling_kwargs={
            "cg_kwargs": {
                "miniter": 0,
                "absdelta": None
            },
            "maxiter": 20,
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
            "energy_reduction_factor": None,
            "absdelta": absdelta,
            "maxiter": n_newton_iterations[i],
            "cg_kwargs": {
                "miniter": 0,
                "name": None
            },
            "name": "N",
        })
    gkl_pos = opt_state.x

# %%
absdelta = 1e-10
opt_state = jft.minimize(
    None,
    x0=jnp.array([1., 1.]),
    method="newton-cg",
    options={
        "fun_and_grad": ham_vg,
        "hessp": ham.metric,
        "energy_reduction_factor": None,
        "absdelta": absdelta,
        "maxiter": 100,
        "cg_kwargs": {
            "miniter": 0,
            "name": None
        },
        "name": "MAP"
    })
map_pos = opt_state.x
key, subkey = random.split(key, 2)
map_geo_samples = GeoMetricKL(
    map_pos,
    100,
    key=subkey,
    mirror_samples=True,
    linear_sampling_name=None,
    linear_sampling_kwargs={"miniter": 0},
    non_linear_sampling_name=None,
    non_linear_sampling_kwargs={
        "cg_kwargs": {
            "miniter": 0
        },
        "maxiter": 20,
    })

# %%

n_pix_sqrt = 1000
x = jnp.linspace(-30 / SCALE, 30 / SCALE, n_pix_sqrt)
y = jnp.linspace(-15 / SCALE, 15 / SCALE, n_pix_sqrt)
X, Y = jnp.meshgrid(x, y)
XY = jnp.array([X, Y]).T
xy = XY.reshape((XY.shape[0] * XY.shape[1], 2))
es = jnp.exp(-lax.map(ham, xy)).reshape(XY.shape[:2]).T

fig, axs = plt.subplots(1, 3, figsize=(16, 9))

b_space_smpls = jnp.array(tuple(mg_samples.at(mkl_pos)))
contour = axs[0].contour(X, Y, es)
axs[0].clabel(contour, inline=True, fontsize=10)
axs[0].scatter(*b_space_smpls.T)
axs[0].plot(*mkl_pos, "rx")
axs[0].set_title("MGVI")

b_space_smpls = jnp.array(tuple(geo_samples.at(gkl_pos)))
contour = axs[1].contour(X, Y, es)
axs[1].clabel(contour, inline=True, fontsize=10)
axs[1].scatter(*b_space_smpls.T, alpha=0.7)
axs[1].plot(*gkl_pos, "rx")
axs[1].set_title("GeoVI")

b_space_smpls = jnp.array(tuple(map_geo_samples.at(map_pos)))
contour = axs[2].contour(X, Y, es)
axs[2].clabel(contour, inline=True, fontsize=10)
axs[2].scatter(*b_space_smpls.T, alpha=0.7)
axs[2].plot(*map_pos, "rx")
axs[2].set_title("MAP + GeoVI Samples")

fig.tight_layout()
fig.savefig("banana_vi_w_regularization.png", dpi=400)
plt.close()
