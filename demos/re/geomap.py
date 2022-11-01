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


# %%
def lanczos_logdet(
    mat,
    v,
    order: int,
):
    """Computes a stochastic estimate of the log-determinate of the Lanczos
    decomposed matrix. This is not the same as applying the stochastic Lanczos
    quadrature algorithm as it estimates the log-determinate for the
    decomposition only.
    """
    mat = mat.__matmul__ if not hasattr(mat, "__call__") else mat

    tridiag, vecs = jft.lanczos.lanczos_tridiag(mat, v, order=order)
    eig_vals = jnp.linalg.eigvalsh(tridiag)
    return jnp.log(eig_vals).sum(), vecs


def _metric_sample(
    hamiltonian: jft.StandardHamiltonian,
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
    prr_inv_metric_smpl = jft.random_like(key=subkey_prr, primals=primals)
    # One may transform any metric sample to a sample of the inverse
    # metric by simply applying the inverse metric to it
    prr_smpl = prr_inv_metric_smpl
    met_smpl = nll_smpl + prr_smpl
    return met_smpl, prr_smpl


def geomap(
    hamiltonian: jft.StandardHamiltonian,
    order: int,
    key,
    sample_orthonormally=True
):
    from jax import flatten_util

    def geomap_energy(pos, return_aux=False):
        p, unflatten = flatten_util.ravel_pytree(pos)

        def mat(x):
            # Hack to stomp arbitrary objects into a 1D array
            o, _ = flatten_util.ravel_pytree(
                hamiltonian.metric(pos, unflatten(x))
            )
            return o

        probe, smpl = _metric_sample(hamiltonian, pos, key)
        probe = flatten_util.ravel_pytree(probe)[0]
        smpl = flatten_util.ravel_pytree(smpl)[0]

        logdet, vecs = lanczos_logdet(mat, probe, order, shape0=p.size)

        if not sample_orthonormally:
            energy = hamiltonian(pos)
            smpl_orig, smpl = None, None
        else:
            #smpl = random.normal(smpl_key, p.shape)
            smpl_orig = unflatten(smpl.copy())
            # TODO: Pull into new lanczos method which computes orthoganlized smpls
            # for vecs
            ortho_smpl = vecs @ smpl
            # One could add an additional `jnp.linalg.inv(vecs @ vecs.T)` in
            # between the vecs to ensure proper projection
            # ortho_smpl = jnp.linalg.inv(vecs @ vecs.T) @ ortho_smpl
            ortho_smpl = vecs.T @ ortho_smpl
            smpl -= ortho_smpl
            smpl = unflatten(smpl)

            # GeoMAP requires the sample to be mirrored as to perform MAP along
            # the subspace in the (near) linear regime. With samples, the
            # solution is not only much less noisy in this regime but is
            # actually the true posterior.
            energy = 0.5 * (hamiltonian(pos + smpl) + hamiltonian(pos - smpl))

        energy += 0.5 * logdet
        if return_aux:
            return energy, (smpl_orig, smpl)
        return energy

    return geomap_energy


# %%
def hartley(p, axes=None):
    from jax.numpy import fft

    tmp = fft.fftn(p, axes)
    return tmp.real + tmp.imag


seed = 42
key = random.PRNGKey(seed)

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
signal_response = lambda x: correlated_field(x)

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

key, subkey, subkey_geomap = random.split(key, 3)
pos_init = jft.random_like(subkey, correlated_field.domain)
pos = 1e-2 * pos_init.copy()

# %%
print("!!! HAM", ham(pos))
print("!!! metric", ham.metric(pos, pos) @ pos)
# This is 50 times slower in compile time than ham.metric
geomap_order = 40
geomap_energy = geomap(
    ham, geomap_order, subkey_geomap, sample_orthonormally=True
)

geomap_energy = jax.jit(geomap_energy, static_argnames=("return_aux", ))
print("!!! geomap_energy", geomap_energy(pos))

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
_, (prr_smpl, ortho_smpl) = geomap_energy(opt_state_geomap.x, return_aux=True)

plt.plot(prr_smpl, label="prior sample", alpha=0.7)
plt.plot(ortho_smpl, label="ortho sample", alpha=0.7)
plt.plot(jnp.abs(prr_smpl - ortho_smpl), label="abs diff", alpha=0.3)
plt.legend()
plt.show()

# %%
smpls_by_order = []
for i in range(1, geomap_order):
    _, (_, s) = geomap(ham, i, subkey_geomap, sample_orthonormally=True)(
        opt_state_geomap.x, return_aux=True
    )
    smpls_by_order += [s]

smpls_by_order = jnp.array(smpls_by_order)
# %%
fig, axs = plt.subplots(2, 1, sharex=True)
d = jnp.diff(smpls_by_order, axis=0)
axs.flat[0].plot(
    smpls_by_order.T, label=jnp.arange(1, geomap_order), alpha=0.3, marker="."
)
axs.flat[0].axhline(0., color="red")
axs.flat[0].legend()
axs.flat[1].plot(
    d.T, label=jnp.arange(1, geomap_order - 1), alpha=0.3, marker="."
)
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
