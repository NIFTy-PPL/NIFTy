#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from collections import namedtuple
from functools import partial
import sys

import jax
from jax import numpy as jnp
from jax import random
from jax.scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kv as mod_bessel2

import nifty8.re as jft

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)

Timed = namedtuple("Timed", ("time", "number"), rename=True)


def timeit(stmt, setup=lambda: None, number=None):
    import timeit

    if number is None:
        number, _ = timeit.Timer(stmt).autorange()

    setup()
    t = timeit.timeit(stmt, number=number) / number
    return Timed(time=t, number=number)


def _matern_kernel(distance, scale, cutoff, dof):
    from jax.scipy.special import gammaln

    reg_dist = jnp.sqrt(2 * dof) * distance / cutoff
    return scale**2 * 2**(1 - dof) / jnp.exp(
        gammaln(dof)
    ) * (reg_dist)**dof * mod_bessel2(dof, reg_dist)


n_dof = 100
n_dist = 1000
min_reg_dist = 1e-6  # approx. lowest resolution of `_matern_kernel` at float64
max_reg_dist = 8e+2  # approx. highest resolution of `_matern_kernel` at float64
eps = 8. * jnp.finfo(jnp.array(min_reg_dist).dtype.type).eps
dof_grid = np.linspace(0., 15., n_dof)
reg_dist_grid = np.logspace(
    np.log(min_reg_dist * (1. - eps)),
    np.log(max_reg_dist * (1. + eps)),
    base=np.e,
    num=n_dist
)
grid = np.meshgrid(dof_grid, reg_dist_grid, indexing="ij")
_unsafe_ln_mod_bessel2 = RegularGridInterpolator(
    (dof_grid, reg_dist_grid), jnp.log(mod_bessel2(*grid)), fill_value=-np.inf
)


def matern_kernel(distance, scale, cutoff, dof):
    from jax.scipy.special import gammaln

    reg_dist = jnp.sqrt(2 * dof) * distance / cutoff
    dof, reg_dist = jnp.broadcast_arrays(dof, reg_dist)

    # Never produce NaNs (https://github.com/google/jax/issues/1052)
    reg_dist = reg_dist.clip(min_reg_dist, max_reg_dist)

    ln_kv = jnp.squeeze(
        _unsafe_ln_mod_bessel2(jnp.stack((dof, reg_dist), axis=-1))
    )
    corr = 2**(1 - dof) * jnp.exp(ln_kv - gammaln(dof)) * (reg_dist)**dof
    return scale**2 * corr


scale, cutoff, dof = 1., 80., 3 / 2

# %%
x = np.logspace(-6, 11, base=jnp.e, num=int(1e+5))
y = _matern_kernel(x, scale, cutoff, dof)
y = jnp.nan_to_num(y, nan=0.)
kernel = partial(jnp.interp, xp=x, fp=y)
kernel_j = partial(matern_kernel, scale=scale, cutoff=cutoff, dof=dof)

fig, ax = plt.subplots()
x_s = x[x < 10 * cutoff]
ax.plot(x_s, kernel(x_s))
ax.plot(x_s, kernel_j(x_s))
ax.plot(x_s, jnp.exp(-(x_s / (2. * cutoff))**2))
ax.set_yscale("log")
fig.savefig("re_refine_kernel.png", transparent=True)
plt.close()

# %%
# Quick demo of the correlated field scheme that is to be used in the following
cf_kwargs = {"shape0": (12, ), "distances0": (50., ), "kernel": kernel}

cf = jft.RefinementField(**cf_kwargs, depth=5)
xi = jft.random_like(random.PRNGKey(42), cf.domain)

fig, ax = plt.subplots(figsize=(8, 4))
for i in range(cf.chart.depth):
    cf_lvl = jft.RefinementField(**cf_kwargs, depth=i)
    x = jnp.mgrid[tuple(slice(sz) for sz in cf_lvl.chart.shape)]
    x = cf.chart.ind2rg(x, i)[0]
    f_lvl = cf_lvl(xi[:i + 1])
    ax.step(x, f_lvl, alpha=0.7, where="mid", label=f"level {i}")
# ax.set_frame_on(False)
# ax.set_xticks([], [])
# ax.set_yticks([], [])
ax.legend()
fig.tight_layout()
fig.savefig("re_refine_field_layers.png", transparent=True)
plt.close()


# %%
def parametrized_kernel(xi, verbose=False):
    scale = jnp.exp(-0.5 + 0.2 * xi["lat_scale"])
    cutoff = jnp.exp(4. + 1e-2 * xi["lat_cutoff"])
    # dof = jnp.exp(0.5 + 0.1 * xi["lat_dof"])
    # kernel = lambda r: xi["scale"] * jnp.exp(-(r / xi["cutoff"])**2)
    if verbose:
        print(f"{scale=}, {cutoff=}, {dof=}")

    return partial(matern_kernel, scale=scale, cutoff=cutoff, dof=dof)


def signal_response(xi):
    return cf(xi["excitations"], parametrized_kernel(xi))


n_std = 0.5

key = random.PRNGKey(45)
key, *key_splits = random.split(key, 4)

xi_truth = jft.random_like(key_splits.pop(), cf.domain)
d = cf(xi_truth, kernel)
d += n_std * random.normal(key_splits.pop(), shape=d.shape)

xi_swd = {
    "excitations": cf.domain,
    "lat_scale": jft.ShapeWithDtype(()),
    "lat_cutoff": jft.ShapeWithDtype(()),
}
pos = 1e-4 * jft.Field(jft.random_like(key_splits.pop(), xi_swd))

n_mgvi_iterations = 15
n_newton_iterations = 15
n_samples = 2
absdelta = 1e-5

nll = jft.Gaussian(d, noise_std_inv=lambda x: x / n_std) @ signal_response
ham = jft.StandardHamiltonian(nll)  # + 0.5 * jft.norm(x, ord=2, ravel=True)
ham_vg = jax.jit(jft.mean_value_and_grad(ham))
ham_metric = jax.jit(jft.mean_metric(ham.metric))
MetricKL = jax.jit(
    partial(jft.MetricKL, ham),
    static_argnames=("n_samples", "mirror_samples", "linear_sampling_name")
)

# %%
# Minimize the potential
for i in range(n_mgvi_iterations):
    print(f"MGVI Iteration {i}", file=sys.stderr)
    print("Sampling...", file=sys.stderr)
    key, subkey = random.split(key, 2)
    samples = MetricKL(
        pos,
        n_samples=n_samples,
        key=subkey,
        mirror_samples=True,
        linear_sampling_kwargs={"absdelta": absdelta / 10.}
    )

    print("Minimizing...", file=sys.stderr)
    opt_state = jft.minimize(
        None,
        x0=pos,
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

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(d, label="data")
ax.plot(cf(xi_truth, kernel), label="truth")
ax.plot(samples.at(pos).mean(signal_response), label="reconstruction")
ax.legend()
fig.tight_layout()
fig.savefig("re_refine_reconstruction.png", transparent=True)
plt.close()

# %%
cf_bench = jft.RefinementField(shape0=(12, ), kernel=kernel, depth=15)
xi_wo = jft.random_like(random.PRNGKey(42), jft.Field(cf_bench.domain))
xi_w = jft.random_like(
    random.PRNGKey(42),
    jft.Field(
        {
            "excitations": cf_bench.domain,
            "lat_scale": jft.ShapeWithDtype(()),
            "lat_cutoff": jft.ShapeWithDtype(()),
        }
    )
)


def signal_response_bench(xi):
    return cf_bench(xi["excitations"], parametrized_kernel(xi))


d = signal_response_bench(0.5 * xi_w)
nll_wo_fwd = jft.Gaussian(d, noise_std_inv=lambda x: x / n_std)
ham_w = jft.StandardHamiltonian(nll_wo_fwd @ signal_response_bench)
ham_wo = jft.StandardHamiltonian(nll_wo_fwd @ cf_bench)

# %%
all_backends = {"cpu"}
all_backends |= {jax.default_backend()}
for backend in all_backends:
    device_kw = {"device": jax.devices(backend=backend)[0]}
    device_put = partial(jax.device_put, **device_kw)

    cf_vag_bench = jax.jit(jax.value_and_grad(ham_w), **device_kw)
    x = device_put(xi_w)
    _ = jax.block_until_ready(cf_vag_bench(x))
    t = timeit(lambda: jax.block_until_ready(cf_vag_bench(x)))
    ti, num = t.time, t.number

    msg = f"{backend.upper()} :: Shape {str(cf_bench.chart.shape):>16s} ({num:6d} loops) :: JAX w/ learnable {ti:4.2e}"
    print(msg, file=sys.stderr)

    cf_vag_bench = jax.jit(jax.value_and_grad(ham_wo), **device_kw)
    x = device_put(xi_wo)
    _ = jax.block_until_ready(cf_vag_bench(x))
    t = timeit(lambda: jax.block_until_ready(cf_vag_bench(x)))
    ti, num = t.time, t.number

    msg = f"{backend.upper()} :: Shape {str(cf_bench.chart.shape):>16s} ({num:6d} loops) :: JAX w/o learnable {ti:4.2e}"
    print(msg, file=sys.stderr)
