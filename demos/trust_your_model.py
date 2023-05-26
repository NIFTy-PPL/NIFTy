#!/usr/bin/env python3

# Copyright(C) 2023 Gordian Edenhofer
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %%
import sys
from functools import partial

import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from jax import random
from jax.config import config
from jax.flatten_util import ravel_pytree

import nifty8.re as jft

config.update("jax_enable_x64", True)

dims = (64, )
cf_zm = {"offset_mean": 0., "offset_std": (1e-3, 1e-4)}
cf_fl = {
    "fluctuations": (1e-1, 5e-3),
    "loglogavgslope": (-1., 1e-2),
    "flexibility": (4e-1, 2e-1),
    "asperity": (2e-1, 5e-2),
    "harmonic_domain_type": "Fourier"
}
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(dims, distances=1. / dims[0], **cf_fl, prefix="ax1")
correlated_field = cfm.finalize()

# %%
seed = 42
key = random.PRNGKey(seed)

signal_response = lambda x: jnp.exp(correlated_field(x))
noise_cov = lambda x: 0.1**2 * x
noise_cov_inv = lambda x: 0.1**-2 * x
noise_std = jnp.sqrt(noise_cov(jnp.ones(correlated_field.target.shape)))

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


@partial(jax.jit, static_argnames=("point_estimates", ))
def sample_evi(primals, key, *, niter, point_estimates=()):
    # at: reset relative position as it gets (wrongly) batched too
    # squeeze: merge "samples" axis with "mirrored_samples" axis
    return jft.smap(
        partial(
            jft.sample_evi,
            nll,
            # linear_sampling_name="S",  # enables verbose logging
            linear_sampling_kwargs={
                "miniter": niter,
                "maxiter": niter
            },
            point_estimates=point_estimates,
        ),
        in_axes=(None, 0)
    )(primals, key).at(primals).squeeze()


# %%
class EyePlusVdaggerV():
    def __init__(self, vecs, xmap=jax.vmap):
        """Linear operator applying :math:`I + V^\\dagger V` to a vector with
        Woodbury inverse.
        """
        self._vecs = vecs
        self._xmap = xmap

    def __matmul__(self, other):
        t = self.xmap(jft.dot, in_axes=(0, None))(self._vecs, other)
        t = jax.tree_map(
            lambda x: self.xmap(jnp.multiply)(x, t).sum(axis=0), self._vecs
        )
        return other + t

    @property
    def xmap(self):
        return self._xmap

    def inv(self):
        return EyePlusVdaggerVInv(self._vecs, xmap=self._xmap)


class EyePlusVdaggerVInv():
    def __init__(self, vecs, xmap=jax.vmap):
        """Sister of `EyePlusVdaggerV` applying the inverse."""
        self._vecs = vecs
        self._xmap = xmap

        self._n_vecs = jax.tree_util.tree_leaves(vecs)[0].shape[0]
        small_outer = xmap(xmap(jft.dot, in_axes=(None, 0)), in_axes=(0, None))
        vvdagger = small_outer(self._vecs, self._vecs)
        self._small = jnp.eye(self._n_vecs) + vvdagger
        self._small_inv = jnp.linalg.inv(self._small)

    @property
    def small(self):
        return self._small

    @property
    def small_inv(self):
        return self._small_inv

    def __matmul__(self, other):
        t = self.xmap(jft.dot, in_axes=(0, None))(self._vecs, other)
        t = self.small_inv @ t  # shape: (n_eff_samples, )
        t = jax.tree_map(
            lambda x: self.xmap(jnp.multiply)(x, t).sum(axis=0), self._vecs
        )
        return other - t

    @property
    def xmap(self):
        return self._xmap


def riemannian_manifold_maximum_a_posterior_and_grad(
    pos,
    data,
    noise_std,
    forward,
    key,
    n_vecs=1,
    mirror_noise=True,
    xmap=jax.vmap,
    _return_trafo_gradient=False,
    _return_vecs=False,
):
    """Riemannian manifold maximum a posteriori and gradient.

    Notes
    -----
    Memory scales quadratically in the number of samples.
    """
    n_eff_samples = 2 * n_vecs if mirror_noise else n_vecs
    samples_key = random.split(key, n_vecs)

    noise_cov_inv = lambda x: x / noise_std**2
    noise_std_inv = lambda x: x / noise_std
    lh_core = partial(
        jft.Gaussian, noise_cov_inv=noise_cov_inv, noise_std_inv=noise_std_inv
    )

    lh = lh_core(data) @ forward
    ham = jft.StandardHamiltonian(lh)

    vecs = []  # TODO: pre-allocate stack of samples
    f = forward(pos)  # TODO combine with ham forward pass
    for i, k in enumerate(np.repeat(samples_key, 1 + mirror_noise, axis=0)):
        n = noise_std * random.normal(k, shape=noise_std.shape)
        d = f + n if i % 2 == 0 else f - n
        lh = lh_core(d) @ forward
        vecs += [jax.grad(lh)(pos)]
    vecs = jax.tree_map(lambda *x: jnp.stack(x), *vecs)
    vecs /= jnp.sqrt(n_eff_samples)
    eye_plus_vdaggerv_inv = EyePlusVdaggerVInv(vecs, xmap=xmap)

    s, ln_det_metric = jnp.linalg.slogdet(eye_plus_vdaggerv_inv.small)
    # assert s == 1
    # print(ln_det_metric)

    grad_ln_det_metric = jft.zeros_like(pos)
    for i, k in enumerate(np.repeat(samples_key, 1 + mirror_noise, axis=0)):
        n = noise_std * random.normal(k, shape=noise_std.shape)
        d = f + n if i % 2 == 0 else f - n
        lh = lh_core(d) @ forward

        nll_smpl = jnp.sqrt(n_eff_samples) * jax.tree_map(lambda x: x[i], vecs)
        grad_ln_det_metric += jft.hvp(
            lh, (pos, ), (eye_plus_vdaggerv_inv @ nll_smpl, )
        )
    grad_ln_det_metric = 2 * grad_ln_det_metric / n_eff_samples

    value, grad = jax.value_and_grad(ham)(pos)
    value += 0.5 * ln_det_metric
    grad += 0.5 * grad_ln_det_metric

    if _return_trafo_gradient and not _return_vecs:
        return value, grad, grad_ln_det_metric
    if _return_vecs and not _return_trafo_gradient:
        return value, grad, vecs
    if _return_trafo_gradient and _return_vecs:
        return value, grad, grad_ln_det_metric, vecs
    return value, grad


# # %%
p = jft.random_like(random.split(key, 99)[-1], correlated_field.domain)
# p = pos_truth

v, g, g_trafo, vecs = jax.vmap(
    partial(
        riemannian_manifold_maximum_a_posterior_and_grad,
        noise_std=noise_std,
        forward=signal_response,
        key=random.split(key, 914)[-1],
        n_vecs=15,
        mirror_noise=True,
        _return_trafo_gradient=True,
        _return_vecs=True,
    ),
    in_axes=(0, None)
)(jft.stack((jft.Vector(p), jft.Vector(p))), data)
pk = (
    "cfax1asperity", "cfax1flexibility", "cfax1fluctuations",
    "cfax1loglogavgslope", "cfzeromode"
)
print({k: g.tree[k] for k in pk})
print({k: g_trafo.tree[k] for k in pk})

# %%
v = jft.stack([ravel_pytree(el)[0] for el in jft.unstack(vecs)], axis=0)
large_outer = jax.vmap(jax.vmap(jnp.dot, in_axes=(None, 1)), in_axes=(1, None))
rmmap_metric = large_outer(v, v)
rmmap_metric += np.eye(len(g))

ham_metric = jax.jit(ham.metric)
probe = jft.zeros_like(correlated_field.domain)
flat_probe, unravel = ravel_pytree(probe)
true_metric = jax.vmap(
    lambda i: ravel_pytree(
        ham_metric(
            jft.Vector(pos_truth), jft.
            Vector(unravel(flat_probe.at[i].set(1.)))
        )
    )[0],
    out_axes=1
)(np.arange(len(flat_probe)))

# %%
true_eigvals = np.linalg.eigvalsh(true_metric)
rmmap_eigvals = np.linalg.eigvalsh(rmmap_metric)

# %%
kw = {
    "vmin":
        min(np.nanmin(np.log(true_metric)), np.nanmin(np.log(rmmap_metric))),
    "vmax":
        np.log(max(true_metric.max(), rmmap_metric.max()))
}
fig, axs = plt.subplots(1, 3, figsize=(8.5, 4))
ax = axs.flat[0]
im = ax.matshow(np.log(true_metric), **kw)
ax.set_title("True Metric")
ax = axs.flat[1]
im = ax.matshow(np.log(rmmap_metric), **kw)
ax.set_title("This Work")
fig.colorbar(im, ax=axs.flat[:2])
ax = axs.flat[2]
ax.plot(true_eigvals[::-1], label="True")
ax.plot(rmmap_eigvals[::-1], label="This Work")
ax.set_yscale("log")
ax.legend()
plt.show()

# %%
n_vecs = 30
n_newton_iterations = 25
n_rmmap_samples = 4
absdelta = 1e-4 * jnp.prod(jnp.array(dims))

key, subkey = random.split(key)
pos_init = jft.random_like(subkey, correlated_field.domain)
pos = 1e-2 * jft.Vector(pos_init.copy())

key, subkey = random.split(key)
_ham_vg = partial(
    riemannian_manifold_maximum_a_posterior_and_grad,
    data=data,
    noise_std=noise_std,
    forward=signal_response,
    key=subkey,
    n_vecs=n_vecs,
    mirror_noise=False,
)
key, sample_key = random.split(key)


@partial(
    partial, key=sample_key, n_samples=n_rmmap_samples, mirror_samples=True
)
def ham_vg(pos, key, n_samples, mirror_samples):
    # NOTE to self, this is wrong! We need to apply the coordinate
    # transfomration to our white samples.
    # TODO: project out the parts that are already covered by the vectors
    samples = jax.vmap(jft.random_like,
                       in_axes=(0, None))(random.split(key, n_samples), pos)
    if mirror_samples:
        samples = jax.tree_map(lambda *x: jnp.concatenate(x), samples, -samples)
    return jax.tree_map(
        partial(jnp.mean, axis=0),
        jax.vmap(_ham_vg, in_axes=(0, ))(pos + samples)
    )


# ham_vg = jax.jit(jax.value_and_grad(ham))
ham_metric = jax.jit(ham.metric)
# TODO: use the actually used metric here (woodbury) and make NCG work with it

# %%  Minimize the potential
print(f"RMMAP Iteration", file=sys.stderr)
opt_state = jft.minimize(
    None,
    pos,
    method="newton-cg",
    options={
        "fun_and_grad": ham_vg,
        "hessp": ham_metric,
        "absdelta": absdelta,
        "maxiter": n_newton_iterations,
        "name": "N",  # enables verbose logging
    }
)
pos = opt_state.x
msg = f"Post RMMAP Iteration: Energy {ham_vg(pos)[0]:2.4e}"
print(msg, file=sys.stderr)

# %%
samples = sample_evi(pos, random.split(key, 5), niter=50)

# %%
namps = cfm.get_normalized_amplitudes()
post_sr_mean = jft.mean(tuple(signal_response(s) for s in samples.at(pos)))
post_a_mean = jft.mean(tuple(cfm.amplitude(s)[1:] for s in samples.at(pos)))
to_plot = [
    ("Signal", signal_response_truth, ""),
    ("Noise", noise_truth, ""),
    ("Data", data, ""),
    ("Reconstruction", post_sr_mean, ""),
    (
        "Ax1",
        (cfm.amplitude(pos_truth)[1:], post_a_mean,
         cfm.amplitude(pos)[1:]), "loglog"
    ),
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
plt.show()
