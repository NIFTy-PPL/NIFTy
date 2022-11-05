#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
import sys

from jax import numpy as jnp
from jax import random
from jax import jit
from jax.config import config
import matplotlib.pyplot as plt

import nifty8.re as jft

config.update("jax_enable_x64", True)

# %%
# ## Likelihood
#
# ### What is a Likelihood in jifty?
#
# * Very generally, the likelihood stores the cost term(s) for the final minimization
#   * P(d|\xi) is a likelihood just like P(\xi) is a likelihood (w/ d := data, \xi := parameters)
#   * Adding two likelihoods yields a likelihood again; thus P(d|\xi) + P(\xi) is just another likelihood
# * Properties
#   * Energy/Hamiltonian: negative log-probability
#   * Left square root (L) of the metric (M; M = L L^\dagger): needed for sampling and minimization
#   * Metric: needed for sampling and minimization; can be inferred from left sqrt metric
#
# ### Differences to NIFTy's `EnergyOperator`?
#
# * There are no operators in jifty, thus there is no EnergyOperator!
# * NIFTy features many different energies classes; in jifty there is just one
# * jifty needs to track the domain of the data without re-introducing operators
#
# ### What gives?
#
# * No manual tracking of the jacobian
# * No linear operators; this also means we can not take the adjoint of the jacobian :(
# * Trivial to define new likelihoods


def Gaussian(data, noise_cov_inv_sqrt):
    # Simple but not very generic Gaussian energy
    def hamiltonian(primals):
        p_res = primals - data
        l_res = noise_cov_inv_sqrt(p_res)
        return 0.5 * jnp.sum(l_res**2)

    def left_sqrt_metric(primals, tangents):
        return noise_cov_inv_sqrt(tangents)

    lsm_tangents_shape = jnp.shape(data)
    # Better: `tree_map(ShapeWithDtype.from_leave, data)`

    return jft.Likelihood(
        hamiltonian,
        left_sqrt_metric=left_sqrt_metric,
        lsm_tangents_shape=lsm_tangents_shape
    )


seed = 42
key = random.PRNGKey(seed)

dims = (1024, )

loglogslope = 2.
power_spectrum = lambda k: 1. / (k**loglogslope + 1.)
modes = jnp.arange((dims[0] / 2) + 1., dtype=float)
harmonic_power = power_spectrum(modes)
harmonic_power = jnp.concatenate((harmonic_power, harmonic_power[-2:0:-1]))

# Specify the model
correlated_field = lambda x: jft.correlated_field.hartley(
    harmonic_power * x.val
)
signal_response = lambda x: jnp.exp(1. + correlated_field(x))
noise_cov_inv_sqrt = lambda x: 0.1**-1 * x

# Create synthetic data
key, subkey = random.split(key)
pos_truth = jft.Field(random.normal(shape=dims, key=key))
signal_response_truth = signal_response(pos_truth)
key, subkey = random.split(key)
noise_truth = 1. / noise_cov_inv_sqrt(jnp.ones(dims)
                                     ) * random.normal(shape=dims, key=key)
data = signal_response_truth + noise_truth

nll = Gaussian(data, noise_cov_inv_sqrt) @ signal_response
ham = jft.StandardHamiltonian(likelihood=nll).jit()
ham_vg = jit(jft.mean_value_and_grad(ham))
ham_metric = jit(jft.mean_metric(ham.metric))
MetricKL = jit(
    partial(jft.MetricKL, ham),
    static_argnames=("n_samples", "mirror_samples", "linear_sampling_name")
)

key, subkey = random.split(key)
pos_init = jft.Field(random.normal(shape=dims, key=subkey))
pos = jft.Field(pos_init.val)

n_newton_iterations = 10
# Maximize the posterior using natural gradient scaling
pos = jft.newton_cg(
    fun_and_grad=ham_vg, x0=pos, hessp=ham_metric, maxiter=n_newton_iterations
)

fig, ax = plt.subplots()
ax.plot(signal_response_truth, alpha=0.7, label="Signal")
ax.plot(noise_truth, alpha=0.7, label="Noise")
ax.plot(data, alpha=0.7, label="Data")
ax.plot(signal_response(pos), alpha=0.7, label="Reconstruction")
ax.legend()
fig.tight_layout()
fig.savefig("n2f_known_spectrum_MAP.png", dpi=400)
plt.close()

# ## Sampling
#
# ### How sampling works in jifty?
#
# To sample from a likelihood, we need to be able to draw samples which have
# the metric as covariance structure and we need to be able to apply the
# inverse metric. The first part is trivial since we can use the left square
# root of the metric associated with every likelihood:
#
#   \tilde{d} \leftarrow \mathcal{G}(0,\mathbb{1})
#   t = L \tilde{d}
#
# with $t$ now having a covariance structure of
#
#   <t t^\dagger> = L <\tilde{d} \tilde{d}^\dagger> L^\dagger = M.
#
# We now need to apply the inverse metric in order to transform the sample to
# an inverse sample. We can do so using the conjugate gradient algorithm which
# yields the solution to $M s = t$, i.e. applies the inverse of $M$ to $t$:
#
#   M s =  t
#   s = M^{-1} t = cg(M, t) .
#
# ### Differences to NIFTy?
#
# * More generic implementation since the left square root of the metric can
#   be applied independently from drawing samples
# * By virtue of storing the left square root metric, no dedicated sampling
#   method needs to be extended ever again
#
# ### What gives?
#
# The clearer separation of sampling and inverting the metric allows for a
# better interplay of our methods with existing tools like JAX's cg
# implementation.

n_mgvi_iterations = 3
n_samples = 4
n_newton_iterations = 5

key, subkey = random.split(key)
pos_init = jft.Field(random.normal(shape=dims, key=subkey))
pos = jft.Field(pos_init.val)

# Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    mg_samples = MetricKL(
        pos,
        n_samples=n_samples,
        key=subkey,
        mirror_samples=True,
    )

    print("Minimizing...", file=sys.stderr)
    pos = jft.newton_cg(
        fun_and_grad=partial(ham_vg, primals_samples=mg_samples),
        x0=pos,
        hessp=partial(ham_metric, primals_samples=mg_samples),
        maxiter=n_newton_iterations
    )
    msg = f"Post MGVI Iteration {i}: Energy {mg_samples.at(pos).mean(ham):2.4e}"
    print(msg, file=sys.stderr)

post_sr_mean = jft.mean(tuple(signal_response(s) for s in mg_samples.at(pos)))
fig, ax = plt.subplots()
ax.plot(signal_response_truth, alpha=0.7, label="Signal")
ax.plot(noise_truth, alpha=0.7, label="Noise")
ax.plot(data, alpha=0.7, label="Data")
ax.plot(post_sr_mean, alpha=0.7, label="Reconstruction")
label = "Reconstructed samples"
for s in mg_samples:
    ax.plot(signal_response(s), color="gray", alpha=0.5, label=label)
    label = None
ax.legend()
fig.tight_layout()
fig.savefig("n2f_known_spectrum_MGVI.png", dpi=400)
plt.close()

# ## Correlated field
#
# ### Correlated fields in jifty
#
# * `CorrelatedFieldMaker` to track amplitudes along different axes
# * `add_fluctuations` method to amend new amplitudes
# * Zero-mode is tracked separately to the amplitudes
# * `finalize` normalizes the amplitudes and takes their outer product
# * Amplitudes are independent of the stack of amplitudes tracked in the correlated field, i.e. no normalization happens within the amplitude
#
# ### Differences to NIFTy
#
# A correlated field with a single axis but arbitrary dimensionality in NIFTy
# is mostly equivalent to one in jifty. Though since jifty does not track
# domains, everything related to harmonic modes and distributing power is
# contained within the correlated field model.
#
# The normalization and factorization of amplitudes is done only once in
# `finalize`. This conceptually simplifies the amplitude model by a lot.
#
# ### What gives?
#
# * Conceptually simpler amplitude model
# * No domains --> no domain mismatches --> broadcasting \o/
# * No domains --> no domain mismatches --> more errors :(

dims_ax1 = (64, )
dims_ax2 = (128, )
cf_zm = {"offset_mean": 0., "offset_std": (1e-3, 1e-4)}
cf_fl = {
    "fluctuations": (1e-1, 5e-3),
    "loglogavgslope": (-1., 1e-2),
    "flexibility": (1e+0, 5e-1),
    "asperity": (5e-1, 1e-1),
    "harmonic_domain_type": "Fourier"
}
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
d = 1. / dims_ax1[0]
cfm.add_fluctuations(dims_ax1, distances=d, **cf_fl, prefix="ax1")
d = 1. / dims_ax2[0]
cfm.add_fluctuations(dims_ax2, distances=d, **cf_fl, prefix="ax2")
correlated_field = cfm.finalize()

signal_response = lambda x: correlated_field(x)
noise_cov = lambda x: 5**2 * x
noise_cov_inv = lambda x: 5**-2 * x

# Create synthetic data
key, subkey = random.split(key)
pos_truth = jft.random_like(subkey, correlated_field.domain)
signal_response_truth = signal_response(pos_truth)
key, subkey = random.split(key)
noise_truth = jnp.sqrt(
    noise_cov(jnp.ones(signal_response_truth.shape))
) * random.normal(shape=signal_response_truth.shape, key=key)
data = signal_response_truth + noise_truth

nll = jft.Gaussian(data, noise_cov_inv) @ signal_response
ham = jft.StandardHamiltonian(likelihood=nll).jit()
ham_vg = jit(jft.mean_value_and_grad(ham))
ham_metric = jit(jft.mean_metric(ham.metric))
MetricKL = jit(
    partial(jft.MetricKL, ham),
    static_argnames=("n_samples", "mirror_samples", "linear_sampling_name")
)

key, subkey = random.split(key)
pos_init = jft.Field(jft.random_like(subkey, correlated_field.domain))
pos = jft.Field(pos_init.val)

n_mgvi_iterations = 3
n_samples = 4
n_newton_iterations = 10

# Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    mg_samples = MetricKL(
        pos,
        n_samples=n_samples,
        key=subkey,
        mirror_samples=True,
    )

    print("Minimizing...", file=sys.stderr)
    pos = jft.newton_cg(
        fun_and_grad=partial(ham_vg, primals_samples=mg_samples),
        x0=pos,
        hessp=partial(ham_metric, primals_samples=mg_samples),
        maxiter=n_newton_iterations
    )
    msg = f"Post MGVI Iteration {i}: Energy {mg_samples.at(pos).mean(ham):2.4e}"
    print(msg, file=sys.stderr)

namps = cfm.get_normalized_amplitudes()
post_sr_mean = jft.mean(tuple(signal_response(s) for s in mg_samples.at(pos)))
post_namps1_mean = jft.mean(tuple(namps[0](s)[1:] for s in mg_samples.at(pos)))
post_namps2_mean = jft.mean(tuple(namps[1](s)[1:] for s in mg_samples.at(pos)))
to_plot = [
    ("Signal", signal_response_truth, "im"),
    ("Noise", noise_truth, "im"),
    ("Data", data, "im"),
    ("Reconstruction", post_sr_mean, "im"),
    ("Ax1", (namps[0](pos_truth)[1:], post_namps1_mean), "loglog"),
    ("Ax2", (namps[1](pos_truth)[1:], post_namps2_mean), "loglog"),
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
fig.savefig("n2f_unknown_factorizing_spectra.png", dpi=400)
plt.close()
