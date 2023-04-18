#!/usr/bin/env python3
# Copyright(C) 2013-2023 Gordian Edenhofer
# SPDX-License-Identifier: BSD-2-Clause

# %%
import sys
from functools import partial

import jax
import matplotlib.pyplot as plt
import nifty8.re as jft
import numpy as np
from jax import numpy as jnp
from jax import random
from jax.config import config

config.update("jax_enable_x64", True)


def get_power_spectrum(cutoff, scale, negloglogslope, high_res_cutoff=1e-3):
    if all(x is None for x in (cutoff, scale, negloglogslope)):
        cutoff, scale, negloglogslope = jnp.exp(
            jnp.array([-6.47776258, 16.43005412, 1.16049258])
        )

    def power_spectrum(k):
        return scale / (1 + (k / cutoff)**
                        negloglogslope) * jnp.exp(-high_res_cutoff * k)

    return power_spectrum


def get_kernel(*args, **kwargs):
    import numpy as np
    from hankel import SymmetricFourierTransform

    ps = get_power_spectrum(*args, **kwargs)

    def kernel_via_hankel(x, dim=3):
        # Hankel's SymmetricFourierTransform can't handle the zero-mode, thus
        # simply add it afterwards.
        ft = SymmetricFourierTransform(ndim=dim, N=30000, h=5e-9)
        Fk = ft.transform(lambda x: np.array(ps(x)), np.array(x), inverse=False)
        return Fk[0] + 1.431e9 / (540 * 740**2)  # add zero mode

    xp = jnp.logspace(-6, 11, 2248, base=jnp.e)
    yp = kernel_via_hankel(xp)
    zm = kernel_via_hankel(0)
    xp = jnp.append(0, xp)
    yp = jnp.log(jnp.append(zm, yp))

    def interp(x):
        return jnp.exp(jnp.interp(x, xp, yp))

    return interp


# %%
# The log of these parameters has been fit on previous reconstructions
kernel_parameters = {
    "cutoff": 1.5372463000446132e-3,
    "scale": 1.3660979564940365e+7,
    "negloglogslope": 3.191504960507704,
    "high_res_cutoff": 1.e-3,
}

cloud = ((170, 180), (-10, -20))  # Taurus
min_shape = (32, 32, 32)

kernel = get_kernel(**kernel_parameters)
r = 300
extent_pos_deg = tuple(np.abs(el[1] - el[0]) for el in cloud)
extent = tuple(np.tan(np.radians(d)) * r for d in extent_pos_deg)
extent += (extent[0], )
distances = tuple(e / s for e, s in zip(extent, min_shape))

cc = jft.CoordinateChart(
    min_shape=min_shape,
    distances=distances,
    irregular_axes=(),
)
rf = jft.RefinementField(cc)


def dust_density_apply(exc, **kw):
    return jnp.exp(rf(exc, **kw))


def projected_dust_density_apply(exc, **kw):
    return dust_density_apply(exc, **kw).sum(axis=-1)


# Make the refinement fast by leaving the kernel fixed
rfm = rf.matrices(kernel)
signal = jft.Model(
    partial(projected_dust_density_apply, kernel=rfm),
    domain=rf.domain,
    init=rf.init
)

# %%
# Generate the Poisson data
key = random.PRNGKey(0)
key, key_signal_truth, key_noise_truth = random.split(key, 3)
signal_truth = signal(signal.init(key_signal_truth))
noise = random.normal(key_noise_truth, signal_truth.shape)
noise *= np.sqrt(signal_truth)
data = signal_truth + noise

# Define the noise solely via the data assuming only Poissonian noise statistics
# and no knowledge about the actual data
noise_bound = np.quantile(data[data > 0], 1e-2)
noise_cov_val = data.clip(noise_bound, None)
noise_cov = lambda x: noise_cov_val * x
noise_cov_inv = lambda x: x / noise_cov_val
nll = jft.Gaussian(data, noise_cov_inv) @ signal

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
im = axs[0].imshow(data, vmin=0., vmax=data.max())
axs[1].imshow(signal_truth, vmin=0., vmax=data.max())
fig.colorbar(im, ax=axs.ravel())
plt.show()


# %%
@partial(jax.jit, static_argnames=("likelihood", ))
def _ham_vg(likelihood, primals, primals_samples):
    assert isinstance(primals_samples, jft.kl.Samples)
    ham = jft.StandardHamiltonian(likelihood=likelihood)
    vvg = jax.vmap(jax.value_and_grad(ham))
    s = vvg(primals_samples.at(primals).samples)
    return jax.tree_util.tree_map(partial(jnp.mean, axis=0), s)


@partial(jax.jit, static_argnames=("likelihood", ))
def _ham_metric(likelihood, primals, tangents, primals_samples):
    assert isinstance(primals_samples, jft.kl.Samples)
    ham = jft.StandardHamiltonian(likelihood=likelihood)
    vmet = jax.vmap(ham.metric, in_axes=(0, None))
    s = vmet(primals_samples.at(primals).samples, tangents)
    return jax.tree_util.tree_map(partial(jnp.mean, axis=0), s)


@partial(jax.jit, static_argnames=("likelihood", ))
def _lh_trafo(likelihood, primals):
    return likelihood.transformation(primals)


@partial(jax.jit, static_argnames=("likelihood", ))
def _lh_lsm(likelihood, primals, tangents):
    return likelihood.left_sqrt_metric(primals, tangents)


def alternating_geoVI(
    likelihood,
    primals,
    key,
    n_samples,
    *,
    n_steps,
    initial_sampling_depth,
    non_linear_sampling_method="NCG",
    nls_kwargs={"maxiter": 1},
):
    draw_metric = partial(
        jft.kl._sample_linearly, likelihood, from_inverse=False
    )
    draw_metric = jax.vmap(draw_metric, in_axes=(None, 0), out_axes=(None, 0))
    draw_samples = partial(
        jft.kl._sample_linearly,
        likelihood,
        from_inverse=True,
        cg_kwargs={"maxiter": initial_sampling_depth}
    )
    draw_samples = jft.smap(draw_samples, in_axes=(None, 0))
    vg = partial(_ham_vg, likelihood)
    metric = partial(_ham_metric, likelihood)
    lh_trafo = partial(_lh_trafo, likelihood)
    lh_lsm = partial(_lh_lsm, likelihood)

    # Initialize the samples
    sample_keys = random.split(key, n_samples)
    smpls, _ = draw_samples(primals, sample_keys)
    smpls = jft.kl.Samples(
        pos=primals,
        samples=jax.tree_map(lambda *x: jnp.concatenate(x), smpls, -smpls)
    )

    # Alternate between minimization and updating the sample
    # TODO: make this the update method of jaxopt style minimzer
    for _ in range(n_steps):
        # Minimize the KL divergence using the current samples
        print("Minimizing...", file=sys.stderr)
        opt_state = jft.minimize(
            None,
            primals,
            method="newton-cg",
            options={
                "fun_and_grad": partial(vg, primals_samples=smpls),
                "hessp": partial(metric, primals_samples=smpls),
                "absdelta": absdelta,
                "maxiter": 1,
                # "name": "N",  # enables verbose logging
            }
        )
        primals = opt_state.x
        # Update the samples non-linearly around the new position. To do so,
        # first update the metric sample to the new position.
        _, met_smpl = draw_metric(primals, sample_keys)
        met_smpl = jft.unstack(
            jax.tree_map(lambda *x: jnp.concatenate(x), met_smpl, -met_smpl)
        )
        # Then curve the samples non-linearly to fit the new position
        print("Curving sample...", file=sys.stderr)
        # TODO: vectorize
        # TODO: put this into a much simpler optimizer e.g. one from jaxopt
        new_smpls = []
        for s, ms in zip(smpls, met_smpl):
            lh_trafo_at_p = lh_trafo(primals)

            def g(x):
                return x - primals + lh_lsm(
                    primals,
                    lh_trafo(x) - lh_trafo_at_p
                )

            r2_half = jft.Gaussian(ms) @ g  # (g - ms)**2 / 2
            opt_state = jft.minimize(
                r2_half,
                x0=s,
                method=non_linear_sampling_method,
                options=nls_kwargs | {"hessp": r2_half.metric},
            )
            new_smpls += [opt_state.x - primals]
            print(f"{opt_state.status}")
        smpls = jft.kl.Samples(pos=primals, samples=jft.stack(new_smpls))

    return primals, smpls


# %%
key, subkey = random.split(key)
pos_init = jft.random_like(subkey, signal.domain)
pos = 1e-2 * jft.Vector(pos_init.copy())

# %%
n_steps = 5
n_samples = 2
absdelta = 1e-4 * jnp.prod(jnp.array(min_shape))
key, subkey = random.split(key)

pos, smpls = alternating_geoVI(
    nll,
    pos,
    subkey,
    n_samples=n_samples,
    n_steps=n_steps,
    initial_sampling_depth=50
)

# %%
fig, axs = plt.subplots(1, 3, figsize=(8, 3), dpi=500)
im = axs[0].imshow(data, vmin=0., vmax=data.max())
axs[1].imshow(signal_truth, vmin=0., vmax=data.max())
axs[2].imshow(
    jax.vmap(signal)(smpls.at(pos).samples).mean(axis=0),
    vmin=0.,
    vmax=data.max()
)
fig.colorbar(im, ax=axs.ravel())
plt.show()
