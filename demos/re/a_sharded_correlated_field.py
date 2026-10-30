#!/usr/bin/env python3
# Copyright(C) 2013-2026 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

# %% [markdown]
# # Sharded correlated fields: ICR and graphgp factors on several devices
#
# A correlated field whose spatial factor is a Gaussian process on a HEALPix
# times log-radius multigrid, generated either by iterative charted refinement
# (`ICRSpace`) or by graphgp on the points of the finest level
# (`GraphGPSpace`), times a harmonic factor along a frequency axis. The field is
# too large for one device in production, so it is *sharded*: every device
# holds a contiguous block of the sky pixels (the leading axis) and all of the
# other axes. The GP models exchange only the halo their refinement needs; the
# harmonic factor and the amplitude models need no code of their own.
#
# The one switch is `finalize(mesh=...)`; when the points of a `GraphGPSpace`
# are already a sharded `jax.Array`, the mesh is inferred from them. The
# excitations drawn by `cf.init` live on the mesh, `cf.shardings` tells where.

# %%
# NOTE: Development only. Four virtual CPU devices so that the demo runs on a
# laptop; remove this on a multi-GPU machine (then also set `cuda=True`).
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import time

import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from jax import random
from jax.sharding import Mesh

import nifty.re as jft

jax.config.update("jax_enable_x64", True)
jft.logger.setLevel("INFO")  # the sharded models report what they infer

cuda = jax.default_backend() == "gpu"  # graphgp's CUDA kernels on GPUs
key = random.PRNGKey(42)

# %% [markdown]
# ## The device mesh
#
# A one-dimensional mesh. The GP models shard the leading axis of their factor
# in contiguous blocks over this one axis: with 4 devices and 768 pixels, each
# device owns 192 pixels. The axis must be divisible by the number of devices
# (for ICR, on the coarse levels that are sharded as well; coarser levels are
# replicated, the model reports the split).

# %%
mesh = Mesh(np.array(jax.devices()), ("x",))
print(mesh)

# %% [markdown]
# ## The grid: HEALPix times log-radius, three refinement levels

# %%
grid = jft.HPLogRGrid(nside=8, nside0=2, r_min_shape=8, r_min=1.0, r_max=10.0)
for lvl in range(grid.depth + 1):
    print(f"level {lvl}: {tuple(grid.at(lvl).shape)}")
n_freq = 16

amplitude = dict(fluctuations=(1.0, 0.1), loglogavgslope=(-3.0, 0.5))
amplitude_freq = dict(fluctuations=(0.5, 0.1), loglogavgslope=(-2.0, 0.5))


def make_field(space, name, mesh=None):
    """Correlated field with the spatial factor `space` times a frequency
    axis; the same amplitude model for both GP variants."""
    cfm = jft.CorrelatedFieldMaker(name)
    cfm.set_amplitude_total_offset(offset_mean=0.0, offset_std=(0.1, 0.01))
    cfm.add_fluctuations(space, prefix="sky", **amplitude)
    cfm.add_fluctuations(jft.RegularSpace((n_freq,), 0.1), prefix="freq", **amplitude_freq)
    return cfm.finalize(mesh=mesh)


# %% [markdown]
# ## An ICR factor
#
# `ICRSpace` holds the multigrid only; the sharding is decided in `finalize`.
# The log tells which levels are replicated (too coarse to split) and which are
# sharded.

# %%
# covariance table shared by both spaces: 3 decades with ~170 bins per decade (ICR needs >= 100 per
# decade), r_min below the smallest separation of the padded coarse levels, r_max above the diagonal
# of the box, and a relative nugget of 0.5 % at zero distance: for steep spectra (slope below about
# -3.2) the tabulated kernel of this amplitude model makes the 3x3x3 refinement windows indefinite
# and the ICR Cholesky returns NaN without it (graphgp's small k x k systems survive either way)
table = dict(r_min=0.02, r_max=40.0, n_bins=512, jitter=5e-3)
cf_icr = make_field(jft.ICRSpace(grid, **table), "icr", mesh=mesh)
print("field shape", cf_icr.target_shape, "sharded as", cf_icr.sharding.spec)
for name, s in cf_icr.shardings.items():
    print(f"  excitations {name!r}:", [x.spec for x in s] if isinstance(s, list) else s.spec)

# %% [markdown]
# ## A graphgp factor on the same grid
#
# `grid_points` gives the coordinates of the finest level as a point set,
# here already placed on the mesh. `GraphGPSpace` keeps the points, the graph
# is built when the field is finalized (on the devices: per-device k-d trees,
# a ring of edge tests against the neighbours' trees, compact per-level tables).
# Without `r_min`/`r_max` the table range would be inferred from the points and
# reported. The mesh is inferred from the sharded points.

# %%
points = jft.grid_points(grid, mesh=mesh)
print("points", points.shape, "sharded as", points.sharding.spec)

t0 = time.time()
cf_ggp = make_field(jft.GraphGPSpace(points, k=8, n0=64, cuda=cuda, **table), "ggp")
print(f"graph build and tables: {time.time() - t0:.1f} s")
print("field shape", cf_ggp.target_shape, "sharded as", cf_ggp.sharding.spec)
for name, s in cf_ggp.shardings.items():
    print(f"  excitations {name!r}:", s.spec)

# %% [markdown]
# ## Samples and gradients stay on the mesh
#
# `cf.init` draws the excitations directly on the devices; under `jit` the
# field and the gradient of any scalar of it come out with the same sharding.
# Nothing is gathered to one device.

# %%
for cf in (cf_icr, cf_ggp):
    key, sub = random.split(key)
    x = cf.init(sub)
    f = jax.jit(cf)(x)
    loss = jax.jit(jax.value_and_grad(lambda x: jnp.sum(cf(x) ** 2)))
    loss(x)  # compile
    t0 = time.time()
    v, g = loss(x)
    jax.block_until_ready(g)
    g_sky = g[f"{cf.prefix}xi"]  # the sky factor's excitations (a list of levels for ICR)
    g_sky = g_sky[-1] if isinstance(g_sky, list) else g_sky
    print(
        f"{cf.prefix}: field {f.shape} on {f.sharding.spec}, gradient of the sky "
        f"excitations on {g_sky.sharding.spec}; value+grad {1e3 * (time.time() - t0):.0f} ms"
    )

# %% [markdown]
# ## The two priors are the same process
#
# Same amplitude model, same points: the two generators realise (approximately)
# the same Gaussian process. Compare the standard deviation per radial bin over
# a few samples and look at a sky sample of each at one radius and frequency.

# %%
n_samples = 4
std = {}
sample = {}
for cf in (cf_icr, cf_ggp):
    fs = []
    for _ in range(n_samples):
        key, sub = random.split(key)
        fs.append(np.asarray(jax.jit(cf)(cf.init(sub))))
    fs = np.stack(fs)
    print(f"{cf.prefix}: {n_samples} samples, finite: {np.isfinite(fs).all()}, std {fs.std():.3f}")
    std[cf.prefix] = fs.std(axis=(0, 1, 3))  # over samples, pixels and frequencies
    sample[cf.prefix] = fs[0]

try:
    import healpy as hp

    fig = plt.figure(figsize=(10, 6.5))
    r = grid.at(grid.depth).shape[1] // 2
    for i, (name, f) in enumerate(sample.items()):
        # the HEALPix grids of nifty.re are in NESTED order (refinement = children of a pixel)
        hp.mollview(f[:, r, n_freq // 2], fig=fig, sub=(2, 2, i + 1), title=f"{name}: sky at radial bin {r}", cmap="RdBu_r", nest=True)
    owner = np.repeat(np.arange(mesh.size), f.shape[0] // mesh.size)
    hp.mollview(owner, fig=fig, sub=(2, 2, 3), title="device owning the pixel", cmap="tab10", min=-0.5, max=9.5, nest=True)
    ax = fig.add_subplot(2, 2, 4)
    for name, s in std.items():
        ax.plot(s, label=name)
    ax.set_xlabel("radial bin")
    ax.set_ylabel("sample std")
    ax.legend()
    fig.subplots_adjust(top=0.94, hspace=0.3)
    fig.savefig("results_sharded_correlated_field.png", dpi=150)
    print("saved results_sharded_correlated_field.png")
except ImportError:
    print("per-bin std:", {k: np.round(v, 2) for k, v in std.items()})

# %% [markdown]
# ## Inference on the mesh
#
# A Gaussian likelihood of noisy observations of the whole field. The data is
# a sharded array as well (it comes out of the sharded model), the likelihood
# reduces it to a scalar, and `optimize_kl` runs as usual: the positions and
# samples are sharded pytrees, every program is sharded by XLA from the
# shardings of its inputs. One iteration with two linear samples, enough to see
# that the machinery runs (a few minutes on emulated CPU devices); the same
# script on four GPUs is the production setup.

# %%
cf = cf_icr  # the ICR field; on CPU the graphgp derivatives (pure-JAX fallback) are much slower
key, k_truth, k_noise, k_init, k_opt = random.split(key, 5)
truth = cf.init(k_truth)
noise_std = 0.5
data = jax.jit(cf)(truth) + noise_std * random.normal(k_noise, cf.target_shape, dtype=jnp.float64)
data = jax.device_put(data, cf.sharding)
lh = jft.Gaussian(data, noise_cov_inv=lambda x: x / noise_std**2).amend(cf)

delta = 1e-4
samples, state = jft.optimize_kl(
    lh,
    jft.Vector(cf.init(k_init)),
    n_total_iterations=1,
    n_samples=2,
    key=k_opt,
    draw_linear_kwargs=dict(cg_name="SL", cg_kwargs=dict(absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=20)),
    kl_kwargs=dict(minimize_kwargs=dict(name="M", xtol=delta, cg_kwargs=dict(name=None, maxiter=10), maxiter=3)),
    sample_mode="linear_resample",
    odir=None,
)
post_mean = jax.jit(cf)(samples.pos)
print("posterior mean on", post_mean.sharding.spec, "rms residual to the truth:", float(jnp.sqrt(jnp.mean((post_mean - jax.jit(cf)(truth)) ** 2))))
