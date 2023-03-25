#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %%
import sys
from functools import partial

import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random
from jax.config import config

import nifty8.re as jft

config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)

dims = (128, 128)

n_mgvi_iterations = 3
n_samples = 4
n_newton_iterations = 10
absdelta = 1e-4 * jnp.prod(jnp.array(dims))

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
cfm.add_fluctuations(dims, distances=1. / dims[0], **cf_fl, prefix="ax1")
correlated_field = cfm.finalize()

# %% [markdown]
# ## Notes on Refinement Field

# The above could cust as well be a refinement field e.g. on a HEALPix sphere
# with logarithmically spaced radial voxels. All of NIFTy.re is agnostic to the
# specifics of the forward model. The sampling and minimization always works the
# same.


# # %%
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
signal_response = lambda x: jnp.exp(correlated_field(x))
noise_cov = lambda x: 0.1**2 * x
noise_cov_inv = lambda x: 0.1**-2 * x

# Create synthetic data
key, subkey = random.split(key)
pos_truth = jft.random_like(subkey, correlated_field.domain)
signal_response_truth = signal_response(pos_truth)
key, subkey = random.split(key)
noise_truth = jnp.sqrt(
    noise_cov(jnp.ones(correlated_field.target.shape))
) * random.normal(shape=correlated_field.target.shape, key=key)
data = signal_response_truth + noise_truth

nll = jft.Gaussian(data, noise_cov_inv) @ signal_response
ham = jft.StandardHamiltonian(likelihood=nll)


@jax.jit
def ham_vg(primals, primals_samples):
    assert isinstance(primals_samples, jft.kl.Samples)
    vvg = jax.vmap(jax.value_and_grad(ham))
    s = vvg(primals_samples.at(primals).samples)
    return jax.tree_util.tree_map(partial(jnp.mean, axis=0), s)


@jax.jit
def ham_metric(primals, tangents, primals_samples):
    assert isinstance(primals_samples, jft.kl.Samples)
    vmet = jax.vmap(ham.metric, in_axes=(0, None))
    s = vmet(primals_samples.at(primals).samples, tangents)
    return jax.tree_util.tree_map(partial(jnp.mean, axis=0), s)


@partial(jax.jit, static_argnames=("point_estimates", ))
def sample_evi(primals, key, *, absdelta, point_estimates=()):
    # at: reset relative position as it gets (wrongly) batched too
    # squeeze: merge "samples" axis with "mirrored_samples" axis
    return jft.smap(
        partial(
            jft.sample_evi,
            nll,
            # linear_sampling_name="S",  # enables verbose logging
            linear_sampling_kwargs={"absdelta": absdelta / 10.},
            point_estimates=point_estimates,
        ),
        in_axes=(None, 0)
    )(primals, key).at(primals).squeeze()


# %%
key, subkey = random.split(key)
pos_init = jft.random_like(subkey, correlated_field.domain)
pos = 1e-2 * jft.Vector(pos_init.copy())

# %%  Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)

    samples = sample_evi(
        pos, random.split(subkey, n_samples), absdelta=absdelta / 10.
    )

    print("Minimizing...", file=sys.stderr)
    opt_state = jft.minimize(
        None,
        pos,
        method="newton-cg",
        options={
            "fun_and_grad": partial(ham_vg, primals_samples=samples),
            "hessp": partial(ham_metric, primals_samples=samples),
            "absdelta": absdelta,
            "maxiter": n_newton_iterations,
            # "name": "N",  # enables verbose logging
        }
    )
    pos = opt_state.x
    msg = f"Post MGVI Iteration {i}: Energy {ham_vg(pos, samples)[0]:2.4e}"
    print(msg, file=sys.stderr)

# %%
namps = cfm.get_normalized_amplitudes()
post_sr_mean = jft.mean(tuple(signal_response(s) for s in samples.at(pos)))
post_a_mean = jft.mean(tuple(cfm.amplitude(s)[1:] for s in samples.at(pos)))
to_plot = [
    ("Signal", signal_response_truth, "im"),
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
