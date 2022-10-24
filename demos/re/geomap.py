#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial
import sys

from jax import numpy as jnp
from jax import random
from jax import jit
import jax
from jax import random
import matplotlib.pyplot as plt

import nifty8.re as jft

jax.config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)

dims = (256, )


def hartley(p, axes=None):
    from jax.numpy import fft

    tmp = fft.fftn(p, axes)
    return tmp.real + tmp.imag


dims = (1024, )

absdelta = 1e-4 * jnp.prod(jnp.array(dims))

cf = {"loglogavgslope": 2.}
loglogslope = cf["loglogavgslope"]
power_spectrum = lambda k: 1. / (k**loglogslope + 1.)

modes = jnp.arange((dims[0] / 2) + 1., dtype=float)
harmonic_power = power_spectrum(modes)
# Every mode appears exactly two times, first ascending then descending
# Save a little on the computational side by mirroring the ascending part
harmonic_power = jnp.concatenate((harmonic_power, harmonic_power[-2:0:-1]))

# Specify the model
correlated_field = jft.Model(
    lambda x: hartley(harmonic_power * x), domain=jft.ShapeWithDtype(dims)
)
signal_response = lambda x: correlated_field(x)  # jnp.exp(1. + correlated_field(x))

noise_cov = lambda x: 0.1**2 * x
noise_cov_inv = lambda x: 0.1**-2 * x

# Create synthetic data
key, subkey = random.split(key)
pos_truth = jft.random_like(subkey, correlated_field.domain)
signal_response_truth = signal_response(pos_truth)
key, subkey = random.split(key)
noise_truth = jnp.sqrt(noise_cov(jnp.ones(dims))
                      ) * random.normal(shape=dims, key=key)
data = signal_response_truth + noise_truth

nll = jft.Gaussian(data, noise_cov_inv) @ signal_response
ham = jft.StandardHamiltonian(likelihood=nll).jit()

plt.plot(jnp.array([signal_response_truth, data]).T, label=("truth", "data"))
plt.legend()
plt.show()


# %%
def _sample_inverse_standard_hamiltonian(
    hamiltonian,
    primals,
    key,
):
    if not isinstance(hamiltonian, jft.StandardHamiltonian):
        te = f"`hamiltonian` of invalid type; got '{type(hamiltonian)}'"
        raise TypeError(te)
    subkey_nll, subkey_prr = random.split(key, 2)
    nll_smpl = jft.kl.sample_likelihood(
        hamiltonian.likelihood, primals, key=subkey_nll
    )
    prr_smpl = jft.random_like(key=subkey_prr, primals=primals)
    met_smpl = nll_smpl + prr_smpl
    return met_smpl, prr_smpl


def stochastic_lq_logdet(
    mat,
    order: int,
    pos,
    key,
    *,
    shape0=None,
):
    """Computes a stochastic estimate of the log-determinate of a matrix using
    the stochastic Lanczos quadrature algorithm.
    """
    shape0 = shape0 if shape0 is not None else mat.shape[0]
    mat = mat.__matmul__ if not hasattr(mat, "__call__") else mat
    if not isinstance(key, jnp.ndarray):
        key = random.PRNGKey(key)
    keys = random.split(key, 1)

    lanczos = jax.tree_util.Partial(
        jft.lanczos.lanczos_tridiag, mat, order=order
    )
    vs, prr_vs = _sample_inverse_standard_hamiltonian(ham, pos, keys[0])
    tridiags, vecs = jax.vmap(lanczos)(jnp.array([vs]))
    return jft.lanczos.stochastic_logdet_from_lanczos(
        tridiags, shape0
    ), vecs[0], prr_vs  # TODO: do not use loop


def geomap(ham: jft.StandardHamiltonian, order: int, key, mirror_samples=True):
    from jax import flatten_util

    def energy(pos, return_sample=False):
        p, unflatten = flatten_util.ravel_pytree(pos)

        def mat(x):
            # Hack to stomp arbitrary objects into a 1D array
            o, _ = flatten_util.ravel_pytree(ham.metric(pos, unflatten(x)))
            return o

        key_lcz, key_smpls = random.split(key, 2)
        logdet, vecs, smpl = stochastic_lq_logdet(
            mat, order, pos, key_lcz, shape0=p.size
        )
        # smpl = random.normal(key_smpls, p.shape, dtype=p.dtype)
        s = smpl.copy()
        # TODO: Pull into new lanczos method which computes orthoganlized smpls
        # for vecs
        ortho_smpl = vecs @ smpl
        # One could add an additional `jnp.linalg.inv(vecs @ vecs.T)` in
        # between the vecs to ensure proper projection
        ortho_smpl = jnp.linalg.inv(vecs @ vecs.T) @ ortho_smpl
        ortho_smpl = vecs.T @ ortho_smpl
        smpl -= ortho_smpl
        smpl = unflatten(smpl)

        if mirror_samples is None:
            h = ham(pos)
        else:
            h = ham(pos + smpl)
            if mirror_samples:
                h += ham(pos - smpl)
                h *= 0.5

        if return_sample:
            return h + 0.5 * logdet, s, smpl
        return h + 0.5 * logdet

    return energy


# %%

key, subkey, subkey_geomap = random.split(key, 3)
pos_init = jft.random_like(subkey, correlated_field.domain)
pos = 1e-2 * pos_init.copy()

# %%
jax.config.update("jax_log_compiles", False)

print("!!!!!!!!!!!!!!!!!!!!!!! HAM", ham(pos))
print("!!!!!!!!!!!!!!!!!!!!!!! metric", ham.metric(pos, pos) @ pos)
# This is 50 times slower in compile time than ham.metric
geomap_order = 5
geomap_energy = geomap(ham, geomap_order, subkey_geomap, mirror_samples=False)

# jft.disable_jax_control_flow._DISABLE_CONTROL_FLOW_PRIM = True
geomap_energy = jax.jit(geomap_energy, static_argnames=("return_sample", ))
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(geomap_energy(pos))

# %%
pos = 1e-2 * pos_init.copy()

opt_state_geomap = jft.minimize(
    geomap_energy,
    pos,
    method="newton-cg",
    options={
        "name": "N",
        "maxiter": 30,
        "cg_kwargs": {
            "name": None
        },
    }
)

# %%
_, prr_smpl, ortho_smpl = geomap_energy(opt_state_geomap.x, return_sample=True)

plt.plot(prr_smpl, label="prior sample", alpha=0.7)
plt.plot(ortho_smpl, label="ortho sample", alpha=0.7)
plt.plot(jnp.abs(prr_smpl - ortho_smpl), label="abs diff", alpha=0.3)
plt.legend()
plt.show()

# %%
smpls_by_order = []
for i in range(1, geomap_order):
    _, _, s = geomap(ham, i, subkey_geomap, mirror_samples=False)(opt_state_geomap.x, return_sample=True)
    smpls_by_order += [s]

smpls_by_order = jnp.array(smpls_by_order)
# %%
fig, axs = plt.subplots(2, 1, sharex=True)
d = jnp.diff(smpls_by_order, axis=0)
axs.flat[0].plot(smpls_by_order.T, label=jnp.arange(1, geomap_order), alpha=0.3, marker=".")
axs.flat[0].axhline(0., color="red")
axs.flat[0].legend()
axs.flat[1].plot(d.T, label=jnp.arange(1, geomap_order - 1), alpha=0.3, marker=".")
axs.flat[1].axhline(0., color="red")
axs.flat[1].legend()
plt.show()

# %%
plt.plot(
    jnp.array(
        [
            signal_response_truth,
            data,
            signal_response(opt_state_geomap.x),
            signal_response(opt_state_geomap.x + ortho_smpl),
        ]
    ).T,
    label=("truth", "data", "rec", "rec + smpl")
)
plt.legend()
plt.show()

# %%
# raise ValueError()

# %%
n_samples = 1
n_newton_iterations = 10
n_mgvi_iterations = 6

ham_vg = jit(jft.mean_value_and_grad(ham))
ham_metric = jit(jft.mean_metric(ham.metric))
MetricKL = jit(
    partial(jft.MetricKL, ham),
    static_argnames=("n_samples", "mirror_samples", "linear_sampling_name")
)

# %%
pos = 1e-2 * pos_init.copy()

# Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    samples = MetricKL(
        pos,
        n_samples=n_samples,
        key=subkey,
        mirror_samples=False,
        linear_sampling_kwargs={
            "absdelta": absdelta / 10.,
            "maxiter": geomap_order
        },
        # linear_sampling_name="S",
    )

    print("Minimizing...", file=sys.stderr)
    opt_state_mgvi = jft.minimize(
        None,
        pos,
        method="newton-cg",
        options={
            "fun_and_grad": partial(ham_vg, primals_samples=samples),
            "hessp": partial(ham_metric, primals_samples=samples),
            "absdelta": absdelta,
            "maxiter": n_newton_iterations
        }
    )
    pos = opt_state_mgvi.x
    msg = f"Post MGVI Iteration {i}: Energy {samples.at(pos).mean(ham):2.4e}"
    print(msg, file=sys.stderr)

# %%
plt.plot(
    jnp.array(
        [
            signal_response_truth,
            data,
            signal_response(opt_state_geomap.x),
            signal_response(opt_state_mgvi.x),
            *samples.at(opt_state_mgvi.x).apply(signal_response),
        ]
    ).T,
    label=(
        "truth",
        "data",
        "rec geomap",
        "rec mgvi",
    ) + ("smpls", ) * len(samples)
)
plt.legend()
plt.show()

# %%
raise ValueError()

# %%

# Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    samples = jft.MetricKL(
        ham,
        pos,
        n_samples=n_samples,
        key=subkey,
        mirror_samples=True,
        linear_sampling_kwargs={"absdelta": absdelta / 10.}
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
            "maxiter": n_newton_iterations
        }
    )
    pos = opt_state.x
    msg = f"Post MGVI Iteration {i}: Energy {samples.at(pos).mean(ham):2.4e}"
    print(msg, file=sys.stderr)

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
