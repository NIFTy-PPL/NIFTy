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

dims = (512, )
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
    key_vecs,
    n_vecs=1,
    mirror_vecs=True,
    key_samples=None,
    n_samples=0,
    mirror_samples=True,
    xmap=jax.vmap,
    _return_trafo_gradient=False,
    _return_vecs=False,
    _return_samples=False,
):
    """Riemannian manifold maximum a posteriori and gradient.

    Notes
    -----
    Memory scales linearly in the number of vectors and samples.
    """
    # TODO: make depndent on dtype of `pos` and `data`
    R_EPS = 1e-13  # sane relative tolerance for vectors with entries ~1

    n_eff_vecs = n_vecs * (1 + mirror_vecs)
    n_eff_samples = n_samples * (1 + mirror_samples)
    key_vecs = random.split(key_vecs, n_vecs)
    key_samples = random.split(key_samples,
                               n_samples) if key_samples is not None else ()
    if n_samples > 0 and key_samples is None:
        ve = "`n_samples` > 0 requires a PRNG key for sampling (`key_samples`)"
        raise ValueError(ve)

    noise_cov_inv = lambda x: x / noise_std**2
    noise_std_inv = lambda x: x / noise_std
    lh_core = partial(
        jft.Gaussian, noise_cov_inv=noise_cov_inv, noise_std_inv=noise_std_inv
    )

    lh = lh_core(data) @ forward
    ham = jft.StandardHamiltonian(lh)

    vecs = []  # TODO: pre-allocate stack of samples
    fwd_at_p = forward(pos)  # TODO combine with ham forward pass
    for i, k in enumerate(jnp.repeat(key_vecs, 1 + mirror_vecs, axis=0)):
        n = noise_std * random.normal(k, shape=noise_std.shape)
        d = (
            fwd_at_p + n if i % 2 == 0 else fwd_at_p - n
        ) if mirror_vecs else fwd_at_p + n
        lh = lh_core(d) @ forward
        vecs += [jax.grad(lh)(pos)]
    vecs = jax.tree_map(lambda *x: jnp.stack(x), *vecs)
    vecs /= jnp.sqrt(n_eff_vecs)
    eye_plus_vdaggerv_inv = EyePlusVdaggerVInv(vecs, xmap=xmap)

    if n_samples > 0:
        samples = xmap(jft.random_like, in_axes=(0, None))(
            jnp.repeat(key_samples, 1 + mirror_samples, axis=0), pos
        )
        s = jnp.tile(jnp.array([1., -1.]) if mirror_samples else 1., n_samples)
        assert s.size == n_eff_samples
        smpl_apply_sgn = partial(xmap(jnp.multiply), s)
        samples = jax.tree_map(smpl_apply_sgn, samples)

        # Create a second orthonormalized stack of vectors for substracting from
        # the samples
        vecs_ortho = jft.zeros_like(vecs)
        for i in range(n_eff_vecs):
            vi = jax.tree_map(lambda x: x[i], vecs)
            for j in range(n_eff_vecs):
                vj = jax.tree_map(lambda x: x[j], vecs_ortho)
                # Effectively truncate the loop by multiplying with (i < j)
                vi -= (j < i) * jft.dot(vi, vj) * vj
            s = jnp.sqrt(jft.dot(vi, vi))
            vi *= jnp.where(s > R_EPS, 1. / s, 0.)
            vecs_ortho = jax.tree_map(
                lambda vecs_o, v: vecs_o.at[i].set(v), vecs_ortho, vi
            )
    else:
        samples = None
        vecs_ortho = None
    # Project out the parts that are already covered by the vectors
    # TODO: vectorize or pull sample creation into loop and vectorize jointly
    for i in range(n_eff_samples):
        si = jax.tree_map(lambda x: x[i], samples)
        f = jax.vmap(jft.dot, in_axes=(0, None))(vecs_ortho, si)
        si -= jax.tree_map(
            partial(jnp.sum, axis=0),
            jax.tree_map(partial(jax.vmap(jnp.multiply), f), vecs_ortho)
        )
        samples = jax.tree_map(lambda smpls, x: smpls.at[i].set(x), samples, si)

    s, ln_det_metric = jnp.linalg.slogdet(eye_plus_vdaggerv_inv.small)
    # assert s == 1
    # print(ln_det_metric)

    grad_ln_det_metric = jft.zeros_like(pos)
    for i, k in enumerate(jnp.repeat(key_vecs, 1 + mirror_vecs, axis=0)):
        n = noise_std * random.normal(k, shape=noise_std.shape)
        d = (
            fwd_at_p + n if i % 2 == 0 else fwd_at_p - n
        ) if mirror_vecs else fwd_at_p + n
        lh = lh_core(d) @ forward

        nll_smpl = jnp.sqrt(n_eff_vecs) * jax.tree_map(lambda x: x[i], vecs)
        grad_ln_det_metric += jft.hvp(
            lh, (pos, ), (eye_plus_vdaggerv_inv @ nll_smpl, )
        )
    grad_ln_det_metric = 2 * grad_ln_det_metric / n_eff_vecs

    if samples is not None:
        # FIXME: account for the gradient of changing samples
        value, grad = xmap(lambda s: jax.value_and_grad(ham)(pos + s))(samples)
        m = partial(jnp.mean, axis=0)
        value, grad = m(value), jax.tree_map(m, grad)
    else:
        value, grad = jax.value_and_grad(ham)(pos)
    value += 0.5 * ln_det_metric
    grad += 0.5 * grad_ln_det_metric

    out = (value, grad)
    if _return_trafo_gradient:
        out += (grad_ln_det_metric, )
    if _return_vecs:
        out += (vecs, )
    if _return_samples:
        out += (samples, )
    return out


# # %%
p = jft.random_like(random.split(key, 99)[-1], correlated_field.domain)
# p = pos_truth

v, g, g_trafo, vecs, samples = partial(
    riemannian_manifold_maximum_a_posterior_and_grad,
    noise_std=noise_std,
    forward=signal_response,
    key_vecs=random.split(key, 914)[-1],
    n_vecs=15,
    mirror_vecs=True,
    key_samples=random.split(key, 922490)[-1],
    n_samples=10,
    mirror_samples=True,
    _return_trafo_gradient=True,
    _return_vecs=True,
    _return_samples=True,
)(jft.Vector(p), data)
pk = (
    "cfax1asperity", "cfax1flexibility", "cfax1fluctuations",
    "cfax1loglogavgslope", "cfzeromode"
)
c = jax.vmap(jax.vmap(jft.dot, in_axes=(0, None)),
             in_axes=(None, 0))(vecs, samples)
print(f"samples are {'' if c.sum() == 0 else 'NOT '}orthonogal to vecs")
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
n_samples = 10
n_newton_iterations = 25
n_rmmap_samples = 4
absdelta = 1e-4 * jnp.prod(jnp.array(dims))

key, subkey = random.split(key)
pos_init = jft.random_like(subkey, correlated_field.domain)
pos = 1e-2 * jft.Vector(pos_init.copy())

key, key_vecs, key_samples = random.split(key, 3)
ham_vg = partial(
    riemannian_manifold_maximum_a_posterior_and_grad,
    data=data,
    noise_std=noise_std,
    forward=signal_response,
    key_vecs=key_vecs,
    n_vecs=n_vecs,
    mirror_vecs=False,
    key_samples=key_samples,
    n_samples=n_samples,
    mirror_samples=True,
)
key, sample_key = random.split(key)

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
# samples = sample_evi(pos, random.split(key, 5), niter=50)

# %%
namps = cfm.get_normalized_amplitudes()
post_sr_mean = signal_response(
    pos
)  # jft.mean(tuple(signal_response(s) for s in samples.at(pos)))
post_a_mean = cfm.amplitude(pos)[
    1:]  # jft.mean(tuple(cfm.amplitude(s)[1:] for s in samples.at(pos)))
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
