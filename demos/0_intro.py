#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %% [markdown]
# # Demonstration of the non-parametric correlated field model in NIFTy.re

# ## The Model

# %%
import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random

import nifty8.re as jft

jax.config.update("jax_enable_x64", True)

seed = 42
key = random.PRNGKey(seed)

dims = (128, 128)

cf_zm = dict(offset_mean=0.0, offset_std=(1e-3, 1e-4))
cf_fl = dict(
    fluctuations=(1e-1, 5e-3),
    loglogavgslope=(-1.0, 1e-2),
    flexibility=(1e0, 5e-1),
    asperity=(5e-1, 5e-2),
)
cfm = jft.CorrelatedFieldMaker("cf")
cfm.set_amplitude_total_offset(**cf_zm)
cfm.add_fluctuations(
    dims, distances=1.0 / dims[0], **cf_fl, prefix="ax1", non_parametric_kind="power"
)
correlated_field = cfm.finalize()

scaling = jft.LogNormalPrior(3.0, 1.0, name="scaling", shape=(1,))


class Signal(jft.Model):
    def __init__(self, correlated_field, scaling):
        self.cf = correlated_field
        self.scaling = scaling
        # Init methods of the Correlated Field model and any prior model in
        # NIFTy.re are aware that their input is standard normal a priori.
        # The `domain` of a model does not know this. Thus, tracking the `init`
        # methods should be preferred over tracking the `domain`.
        super().__init__(init=self.cf.init | self.scaling.init)

    def __call__(self, x):
        # NOTE, think of `Model` as being just a plain function that takes some
        # input and performs all the necessary computation for your model.
        # Note, `scaling` here is completely degenarate with `offset_std` in the
        # likelihood but the priors for them are very different.
        return self.scaling(x) * jnp.exp(self.cf(x))


signal = Signal(correlated_field, scaling)

# %% [markdown]
# ### NIFTy to NIFTy.re

# The equivalent model for the correlated field in numpy-based NIFTy reads

# ```python
# import nifty8 as ift
#
# position_space = ift.RGSpace(dims, distances=1. / dims[0])
# cf_fl_nft = {
#     k: v
#     for k, v in cf_fl.items() if k not in ("harmonic_domain_type", )
# }
# cfm_nft = ift.CorrelatedFieldMaker("cf")
# cfm_nft.add_fluctuations(position_space, **cf_fl_nft, prefix="ax1")
# cfm_nft.set_amplitude_total_offset(**cf_zm)
# correlated_field_nft = cfm_nft.finalize()
# ```

# For convience, NIFTy implements a method to translate numpy-based NIFTy
# operators to NIFTy.re. One can access the equivalent expression in JAX for a
# NIFTy model via the `.jax_expr` property of an operator. In addition, NIFTy
# features a method to additionally preserve the domain and target:
# `ift.nifty2jax.convert` translate NIFTy operators to `jft.Model`. NIFTy.re
# models feature `.domain` and `.target` properties but instead of yielding
# domains, they return [JAX PyTrees](TODO:cite PyTree docu) of shape-and-dtype
# objects.

# ```python
# # Convenience method to get JAX expression as NIFTy.re model which tracks
# # domain and target
# correlated_field_nft: jft.Model = ift.nifty2jax.convert(
#     correlated_field_nft, float
# )
# ```

# Both expressions are identical up to floating point precision
# ```python
# import numpy as np
#
# t = correlated_field_nft.init(random.PRNGKey(42))
# np.testing.assert_allclose(
#     correlated_field(t), correlated_field_nft(t), atol=1e-13, rtol=1e-13
# )
# ```

# Note, caution is advised when translating NIFTy models working on complex
# numbers. Numyp-based NIFTy models are not dtype aware and thus require more
# care when translating them to NIFTy.re/JAX which requires known dtypes.

# %% [markdown]
# ### Notes on Refinement Field

# The above could just as well be a refinement field e.g. on a HEALPix sphere
# with logarithmically spaced radial voxels. All of NIFTy.re is agnostic to the
# specifics of the forward model. The sampling and minimization always works the
# same.

# ```python
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
# ```

# %% [markdown]
# ## The likelihood

# %%
signal_response = signal
noise_cov = lambda x: 0.1**2 * x
noise_cov_inv = lambda x: 0.1**-2 * x

# Create synthetic data
key, subkey = random.split(key)
pos_truth = jft.random_like(subkey, signal_response.domain)
signal_response_truth = signal_response(pos_truth)
key, subkey = random.split(key)
noise_truth = (
    (noise_cov(jft.ones_like(signal_response.target))) ** 0.5
) * jft.random_like(key, signal_response.target)
data = signal_response_truth + noise_truth

lh = jft.Gaussian(data, noise_cov_inv).amend(signal_response)

# %% [markdown]
# ## The inference

# %%
n_vi_iterations = 6
delta = 1e-4
n_samples = 4

key, k_i, k_o = random.split(key, 3)
# NOTE, changing the number of samples always triggers a resampling even if
# `resamples=False`, as more samples have to be drawn that did not exist before.
samples, state = jft.optimize_kl(
    lh,
    jft.Vector(lh.init(k_i)),
    n_total_iterations=n_vi_iterations,
    n_samples=lambda i: n_samples // 2 if i < 2 else n_samples,
    # Source for the stochasticity for sampling
    key=k_o,
    # Names of parameters that should not be sampled but still optimized
    # can be specified as point_estimates (effectively we are doing MAP for
    # these degrees of freedom).
    # point_estimates=("cfax1flexibility", "cfax1asperity"),
    # Arguments for the conjugate gradient method used to drawing samples from
    # an implicit covariance matrix
    draw_linear_kwargs=dict(
        cg_name="SL",
        cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=100),
    ),
    # Arguements for the minimizer in the nonlinear updating of the samples
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None),
            maxiter=5,
        )
    ),
    # Arguments for the minimizer of the KL-divergence cost potential
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name="M", xtol=delta, cg_kwargs=dict(name=None), maxiter=35
        )
    ),
    sample_mode="nonlinear_resample",
    odir="results_intro",
    resume=False,
)

# %%
namps = cfm.get_normalized_amplitudes()
post_sr_mean = jft.mean(tuple(signal(s) for s in samples))
post_a_mean = jft.mean(tuple(cfm.amplitude(s)[1:] for s in samples))
grid = correlated_field.target_grids[0]
to_plot = [
    ("Signal", signal(pos_truth), "im"),
    ("Noise", noise_truth, "im"),
    ("Data", data, "im"),
    ("Reconstruction", post_sr_mean, "im"),
    (
        "Amplitude spectrum",
        (
            grid.harmonic_grid.mode_lengths[1:],
            cfm.amplitude(pos_truth)[1:],
            post_a_mean,
        ),
        "loglog",
    ),
]
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
for ax, v in zip(axs.flat, to_plot):
    title, field, tp, *labels = v
    ax.set_title(title)
    if tp == "im":
        end = tuple(n * d for n, d in zip(grid.shape, grid.distances))
        im = ax.imshow(field.T, cmap="inferno", extent=(0.0, end[0], 0.0, end[1]))
        plt.colorbar(im, ax=ax, orientation="horizontal")
    else:
        ax_plot = ax.loglog if tp == "loglog" else ax.plot
        x = field[0]
        for f in field[1:]:
            ax_plot(x, f, alpha=0.7)
for ax in axs.flat[len(to_plot) :]:
    ax.set_axis_off()
fig.tight_layout()
fig.savefig("results_intro_full_reconstruction.png", dpi=400)
plt.show()
