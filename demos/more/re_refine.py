#!/usr/bin/env python3

# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from collections import namedtuple
from functools import partial
import sys

import jax
from jax import numpy as jnp
from jax import random
from jax import config as jax_config
from jax import vmap
from jax.scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kv as mod_bessel2

import nifty8.re as jft
from nifty8.re import refine

jax_config.update("jax_enable_x64", True)
# jax_config.update("jax_debug_nans", True)
interactive = False

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

if interactive:
    fig, ax = plt.subplots()
    x_s = x[x < 10 * cutoff]
    ax.plot(x_s, kernel(x_s))
    ax.plot(x_s, kernel_j(x_s))
    ax.plot(x_s, jnp.exp(-(x_s / (2. * cutoff))**2))
    ax.set_yscale("log")
    plt.show()

# %%
# Quick demo of the correlated field scheme that is to be used in the following
cf, exc_shp = refine.get_fixed_power_correlated_field(
    shape0=(12, 12), distances0=(200., 200.), depth=5, kernel=kernel
)

if interactive:
    fig, ax = plt.subplots()
    xi = jft.random_like(random.PRNGKey(42), exc_shp)
    im = ax.imshow(cf(xi))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()

# %%
coord = jnp.linspace(0., 500., num=500)
coord = coord.reshape(1, -1)

kern_outer = vmap(
    vmap(lambda x, y: kernel(jnp.linalg.norm(x - y)), in_axes=(None, 1)),
    in_axes=(1, None)
)

key = random.PRNGKey(42)
key, k_c, k_f = random.split(key, 3)

main_coord = coord[:, ::10]
coarse_coord = main_coord[:, :3]
fine_coord = coarse_coord[tuple(jnp.array(coarse_coord.shape) // 2)
                         ] + (jnp.diff(coarse_coord) / jnp.array([-4., 4.]))
cov_ff = kern_outer(fine_coord, fine_coord)
cov_fc = kern_outer(fine_coord, coarse_coord)
cov_cc_inv = jnp.linalg.inv(kern_outer(coarse_coord, coarse_coord))

fine_kernel = cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
# For falling power spectra the above must theoretically always have a positive
# diagonal
if jnp.all(jnp.diag(fine_kernel) > 0.):
    fine_kernel_sqrt = jnp.linalg.cholesky(fine_kernel)
else:
    fine_kernel_sqrt = jnp.diag(jnp.sqrt(jnp.abs(jnp.diag(fine_kernel))))
olf = cov_fc @ cov_cc_inv

if interactive:
    fig, axs = plt.subplots(1, 3)
    im = axs.flat[0].matshow(kern_outer(coord, coord))
    fig.colorbar(im, ax=axs.flat[0])
    im = axs.flat[1].matshow(olf)
    fig.colorbar(im, ax=axs.flat[1])
    im = axs.flat[2].matshow(fine_kernel_sqrt)
    fig.colorbar(im, ax=axs.flat[2])
    plt.show()

# %%
# N - batch dimension
# H - spatial height
# W - spatial width
# D - spatial depth
# C - channel dimension
# I - kernel input channel dimension
# O - kernel output channel dimension
#dim_nums = ('NCWH', 'IWHO', 'NWHC')

cov_sqrt = jnp.linalg.cholesky(kern_outer(main_coord, main_coord))
lvl0 = cov_sqrt @ random.normal(k_c, shape=main_coord.shape[::-1])

refined_m = vmap(
    partial(jnp.convolve, mode="valid"), in_axes=(None, 0), out_axes=1
)(lvl0.ravel(), olf[::-1])

lvl1_exc = random.normal(k_f, shape=refined_m.shape)
refined_std = vmap(jnp.matmul, in_axes=(None, 0))(fine_kernel_sqrt, lvl1_exc)
lvl1 = (refined_m + refined_std).ravel()

lvl1_full_coord = main_coord[
    ..., np.newaxis] + jnp.diff(coarse_coord) / jnp.array([-4., 4.])
lvl1_full_coord = lvl1_full_coord[:, 1:-1, :].ravel()
if interactive:
    plt.plot(main_coord.ravel(), lvl0)
    plt.plot(lvl1_full_coord, lvl1)
    plt.show()

# %%
distances0 = 10.
distances1 = distances0 / 2
distances2 = distances1 / 2

key, k_c, *k_f = random.split(random.PRNGKey(42), 6)

main_coord = jnp.linspace(0., 200., 25)
cov_from_loc = vmap(
    vmap(lambda x, y: kernel(jnp.linalg.norm(x - y)), in_axes=(None, 0)),
    in_axes=(0, None)
)
cov_sqrt = jnp.linalg.cholesky(cov_from_loc(main_coord, main_coord))
lvl0 = cov_sqrt @ random.normal(k_c, shape=main_coord.shape[::-1])

lvl1_exc = random.normal(k_f[1], shape=(2 * (lvl0.size - 2), ))
lvl1 = refine.refine(
    lvl0.ravel(), lvl1_exc,
    *refine.layer_refinement_matrices(distances0, kernel)
)
lvl1_wo_exc = refine.refine(
    lvl0.ravel(), 0. * lvl1_exc,
    *refine.layer_refinement_matrices(distances0, kernel)
)
lvl2_exc = random.normal(k_f[2], shape=(2 * (lvl1.size - 2), ))
lvl2 = refine.refine(
    lvl1.ravel(), lvl2_exc,
    *refine.layer_refinement_matrices(distances1, kernel)
)
lvl3_exc = random.normal(k_f[3], shape=(2 * (lvl2.size - 2), ))
lvl3 = refine.refine(
    lvl2.ravel(), lvl3_exc,
    *refine.layer_refinement_matrices(distances2, kernel)
)

if interactive:
    x0 = jnp.arange(0., jnp.size(lvl0), dtype=float) * distances0
    x1 = jnp.linspace(
        x0.min() + 0.75 * distances0,
        x0.max() - 0.75 * distances0, lvl1.size
    )
    x2 = jnp.linspace(
        x1.min() + 0.75 * distances1,
        x1.max() - 0.75 * distances1, lvl2.size
    )
    x3 = jnp.linspace(
        x2.min() + 0.75 * distances2,
        x2.max() - 0.75 * distances1, lvl3.size
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(x0, lvl0.ravel(), alpha=0.7, where="mid", label="LVL0")
    ax.step(
        x1, lvl1_wo_exc, alpha=0.7, where="mid", label="LVL1 w/o Excitations"
    )
    ax.step(x1, lvl1, alpha=0.7, where="mid", label="LVL1")
    ax.step(x2, lvl2, alpha=0.7, where="mid", label="LVL2")
    ax.step(x3, lvl3, alpha=0.7, where="mid", label="LVL3")
    ax.set_frame_on(False)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.legend()
    fig.tight_layout()
    plt.show()

# %%
key = random.PRNGKey(45)

distances0 = jnp.array([6e+2, 6e+2])
n_std = 0.5

# Manually toggle the following values
depth = 4
coarse_sz = 5
with_zeros = True
shape0 = (12, 12)

n_plots = 9

fig, axs = plt.subplots(*((int(n_plots**0.5), ) * 2), figsize=(16, 16))
for i in range(n_plots):
    key, _ = random.split(key, 2)

    xi_truth_swd = refine.get_refinement_shapewithdtype(
        shape0, depth, _coarse_size=coarse_sz
    )
    xi = jft.random_like(key, xi_truth_swd)

    os, (cov_sqrt0, ks) = refine.refinement_matrices(
        shape0,
        depth,
        distances0=distances0,
        kernel=kernel,
        _coarse_size=coarse_sz,
        _with_zeros=with_zeros,
    )

    fine = (cov_sqrt0 @ xi[0].ravel()).reshape(xi[0].shape)
    for x, olf, k in zip(xi[1:], os, ks):
        fine = refine.refine(fine, x, olf, k, _coarse_size=coarse_sz)

    im = axs.flat[i].imshow(fine)
    fig.colorbar(im, ax=axs.flat[i])
fig.tight_layout()
plt.show()

# %%
depth = 4
shape0 = (12, 12)
xi_truth_swd = refine.get_refinement_shapewithdtype(shape0, depth)
xi = jft.random_like(random.PRNGKey(42), xi_truth_swd)

os, (cov_sqrt0, ks) = refine.refinement_matrices(
    shape0,
    depth,
    distances0=distances0,
    kernel=kernel,
    _with_zeros=True,
)

fine = (cov_sqrt0 @ xi[0].ravel()).reshape(xi[0].shape)
for x, olf, k in zip(xi[1:], os, ks):
    fine = refine.refine(fine, x, olf, k)

fine_w0 = fine.copy()

os, (cov_sqrt0, ks) = refine.refinement_matrices(
    shape0, depth, distances0=distances0, kernel=kernel
)

fine = (cov_sqrt0 @ xi[0].ravel()).reshape(xi[0].shape)
for x, olf, k in zip(xi[1:], os, ks):
    fine = refine.refine(fine, x, olf, k)

mi, ma = min(fine_w0.min(), fine.min()), max(fine_w0.max(), fine.max())
fig, axs = plt.subplots(1, 3)
im = axs.flat[0].imshow(fine_w0.clip(mi, ma))
fig.colorbar(im, ax=axs.flat[0])
im = axs.flat[1].imshow(fine.clip(mi, ma))
fig.colorbar(im, ax=axs.flat[1])
im = axs.flat[2].imshow(fine - fine_w0)
fig.colorbar(im, ax=axs.flat[2])
plt.show()
# %%

fig, axs = plt.subplots(1, 2)
f = np.fft.fft2(fine)
axs.flat[0].imshow(f.real)
axs.flat[1].imshow(f.imag)
plt.show()

# %%
distances0 = jnp.array([5e+2, 3e+2])
n_std = 0.5

depth = 3
shape0 = (12, 12)
xi_swd = refine.get_refinement_shapewithdtype(shape0, depth)

ds = []
for seed in range(100):
    xi = jft.random_like(random.PRNGKey(seed), xi_swd)
    ds += [refine.correlated_field(xi, distances0, kernel).ravel()]
ds = np.stack(ds, axis=1)

plt.imshow(np.cov(ds))
plt.colorbar()
plt.show()

# %%
key = random.PRNGKey(45)
key, *key_splits = random.split(key, 4)

xi_truth_swd = refine.get_refinement_shapewithdtype(shape0, depth)
xi_truth = jft.random_like(key_splits.pop(), xi_truth_swd)
d = refine.correlated_field(xi_truth, distances0, kernel)
d += n_std * random.normal(key_splits.pop(), shape=d.shape)


def signal_response(xi, distances0):
    xi = jft.Field(xi.val.copy())
    # Un-standardize parameters
    xi.val["scale"] = jnp.exp(-0.5 + 0.2 * xi.val.pop("lat_scale"))
    xi.val["cutoff"] = jnp.exp(4. + 1e-2 * xi.val.pop("lat_cutoff"))
    xi.val["dof"] = jnp.exp(0.5 + 0.1 * xi.val.pop("lat_dof"))
    kernel = partial(
        matern_kernel, scale=xi["scale"], cutoff=xi["cutoff"], dof=xi["dof"]
    )

    # xi.val["scale"] = jnp.exp(-0.1 + 0.1 * xi.val.pop("lat_scale"))
    # xi.val["cutoff"] = jnp.exp(4.5 + 0.1 * xi.val.pop("lat_cutoff"))
    # kernel = lambda r: xi["scale"] * jnp.exp(-(r / xi["cutoff"])**2)

    return refine.correlated_field(xi["excitations"], distances0, kernel)


xi_swd = {
    "excitations": xi_truth_swd,
    "lat_scale": jft.ShapeWithDtype(()),
    "lat_cutoff": jft.ShapeWithDtype(()),
    "lat_dof": jft.ShapeWithDtype(())
}
xi = 1e-4 * jft.Field(jft.random_like(key_splits.pop(), xi_swd))
ham = lambda x: jnp.linalg.norm(d - signal_response(x, distances0), ord=2) / (
    2 * n_std**2
) + 0.5 * jft.norm(x, ord=2, ravel=True)
ham = jax.jit(ham)

if interactive and d.ndim == 1:
    plt.plot(d, label="Data")
    plt.plot(
        refine.correlated_field(xi_truth, distances0, kernel), label="Truth"
    )
    plt.plot(signal_response(xi, distances0), label="Start")
    plt.legend()
    plt.show()
elif interactive and d.ndim == 2:
    fig, axs = plt.subplots(1, 3)
    axs.flat[0].imshow(d)
    axs.flat[1].imshow(refine.correlated_field(xi_truth, distances0, kernel))
    axs.flat[2].imshow(signal_response(xi, distances0))
    plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(
    refine.correlated_field(xi_truth, distances0, kernel).T, cmap="Blues"
)
# fig.colorbar(im)
ax.set_frame_on(False)
ax.set_xticks([], [])
ax.set_yticks([], [])
fig.tight_layout()
plt.show()

# %%
if interactive:
    opt_state = jft.minimize(
        ham,
        xi,
        method="NewtonCG",
        options={
            "energy_reduction_factor": None,
            "absdelta": 0.1,
            "maxiter": 30,
            "cg_kwargs": {
                "miniter": 0,
                "name": "NCG",
            },
            "name": "N",
        }
    )

    if d.ndim == 1:
        plt.plot(d, label="Data")
        plt.plot(
            refine.correlated_field(xi_truth, distances0, kernel),
            label="Truth"
        )
        plt.plot(signal_response(xi, distances0), label="Start")
        plt.plot(signal_response(opt_state.x, distances0), label="Final")
        plt.legend()
        plt.show()
    elif d.ndim == 2:
        fig, axs = plt.subplots(1, 4)
        axs.flat[0].imshow(d)
        axs.flat[1].imshow(
            refine.correlated_field(xi_truth, distances0, kernel)
        )
        axs.flat[2].imshow(signal_response(xi, distances0))
        axs.flat[3].imshow(signal_response(opt_state.x, distances0))
        for ax in axs:
            ax.set_frame_on(False)
            ax.set_xticks([], [])
            ax.set_yticks([], [])
        plt.show()

# %%
# JIFTy CF :: Shape        (262144,) (    10 loops) :: JAX 2.83e-02 :: NIFTy 3.93e-02
# # # GRID REFINEMENT (shape0 = 12) CPU: 8 cores 22 GB; GPU: A100 40 GB (refine_conv_general)
# GRID REFINEMENT :: CPU :: Shape        (262148,) (    20 loops) :: JAX w/ learnable 1.13e-02
# GRID REFINEMENT :: CPU :: Shape        (262148,) (    50 loops) :: JAX w/o learnable 8.91e-03
# GRID REFINEMENT :: GPU :: Shape        (262148,) (    50 loops) :: JAX w/ learnable 9.62e-03
# GRID REFINEMENT :: GPU :: Shape        (262148,) (   100 loops) :: JAX w/o learnable 2.78e-03
# # # GRID REFINEMENT (shape0 = 12) CPU: 8 cores 22 GB; GPU: A100 40 GB (refine_vmap)
# GRID REFINEMENT :: CPU :: Shape        (262148,) (    20 loops) :: JAX w/ learnable 1.09e-02
# GRID REFINEMENT :: CPU :: Shape        (262148,) (    50 loops) :: JAX w/o learnable 8.08e-03
# GRID REFINEMENT :: GPU :: Shape        (262148,) (    50 loops) :: JAX w/ learnable 8.95e-03
# GRID REFINEMENT :: GPU :: Shape        (262148,) (   100 loops) :: JAX w/o learnable 2.21e-03
# # # GRID REFINEMENT (shape0 = 12) CPU: 8 cores 22 GB; GPU: A100 40 GB (refine_conv)
# GRID REFINEMENT :: CPU :: Shape        (262148,) (    50 loops) :: JAX w/ learnable 5.80e-03
# GRID REFINEMENT :: CPU :: Shape        (262148,) (   100 loops) :: JAX w/o learnable 3.80e-03
# GRID REFINEMENT :: GPU :: Shape        (262148,) (    10 loops) :: JAX w/ learnable 2.34e-02
# GRID REFINEMENT :: GPU :: Shape        (262148,) (   200 loops) :: JAX w/o learnable 1.44e-03
# # # GRID REFINEMENT (shape0 = 12) CPU: 8 cores 22 GB; GPU: A100 40 GB (refine_loop)
# GRID REFINEMENT :: CPU :: Shape        (262148,) (    50 loops) :: JAX w/ learnable 6.19e-03
# GRID REFINEMENT :: CPU :: Shape        (262148,) (    50 loops) :: JAX w/o learnable 4.55e-03
# GRID REFINEMENT :: GPU :: Shape        (262148,) (    50 loops) :: JAX w/ learnable 8.23e-03
# GRID REFINEMENT :: GPU :: Shape        (262148,) (   200 loops) :: JAX w/o learnable 1.66e-03
all_backends = ["cpu"]
all_backends += [jax.default_backend()
                ] if jax.default_backend() != "cpu" else []
for backend in all_backends:
    device_kw = {"device": jax.devices(backend=backend)[0]}
    device_put = partial(jax.device_put, **device_kw)

    corr = jax.jit(
        jax.value_and_grad(
            lambda x: ((1. - signal_response(x, distances0))**2).sum()
        ), **device_kw
    )
    x = xi

    x = device_put(x)
    _ = corr(x)[0].block_until_ready()
    t = timeit(lambda: corr(x)[0].block_until_ready())
    ti, num = t.time, t.number

    lvln1_shp = xi_truth_swd[-1].shape
    print(
        f"{backend.upper()} :: Shape {str(lvln1_shp):>16s} ({num:6d} loops) :: JAX w/ learnable {ti:4.2e}",
        file=sys.stderr
    )

    corr = jax.jit(
        jax.value_and_grad(
            lambda x:
            ((1. - refine.correlated_field(x, distances0, kernel))**2).sum()
        ), **device_kw
    )
    x = xi_truth

    x = device_put(x)
    _ = corr(x)[0].block_until_ready()
    t = timeit(lambda: corr(x)[0].block_until_ready())
    ti, num = t.time, t.number

    print(
        f"{backend.upper()} :: Shape {str(lvln1_shp):>16s} ({num:6d} loops) :: JAX w/o learnable {ti:4.2e}",
        file=sys.stderr
    )

# %%
coarse_values = refine.correlated_field(xi_truth, distances0, kernel)
olf, fine_kernel_sqrt = refine.layer_refinement_matrices(distances0, kernel)

cv = coarse_values
exc = random.normal(
    key,
    shape=tuple(n - 2 for n in coarse_values.shape) + (2, ) * len(distances0)
)
ref = jax.jit(refine.refine)
_ = ref(cv, exc, olf, fine_kernel_sqrt)
timeit(lambda: ref(cv, exc, olf, fine_kernel_sqrt).block_until_ready())


# %%
def fwd_diy(xi, opt_lin_filter, kernel_sqrt):
    cov_sqrt0, ks = kernel_sqrt
    fine = (cov_sqrt0 @ xi[0].ravel()).reshape(xi[0].shape)
    for x, olf, k in zip(xi[1:], opt_lin_filter, ks):
        fine = refine.refine(fine, x, olf, k)
    return fine


x = jft.random_like(key, jft.Field(xi_truth_swd))
olfs, kss = refine.refinement_matrices(
    x.val[0].shape, len(x.val), distances0, kernel
)
corr = jax.jit(jax.value_and_grad(lambda x: fwd_diy(x, olfs, kss).sum()), )

_ = corr(x)[0].block_until_ready()
timeit(lambda: corr(x)[0].block_until_ready())

# %%
rm = jax.jit(refine.layer_refinement_matrices, static_argnames=("kernel", ))
_ = rm(distances0, kernel)
timeit(lambda: rm(distances0, kernel)[0].block_until_ready())

# %%
grm = jax.jit(
    refine.refinement_matrices, static_argnames=(
        "shape0",
        "depth",
        "kernel",
    )
)
_ = grm(x.val[0].shape, len(x.val), distances0, kernel)
timeit(
    lambda: grm(x.val[0].shape, len(x.val), distances0, kernel)[0].
    block_until_ready()
)

# %%
cv = coarse_values.copy()
exc = random.normal(
    key,
    shape=tuple(n - 2 for n in coarse_values.shape) + (2, ) * len(distances0)
)
olf, ks = refine.layer_refinement_matrices(distances0, kernel)
ref = jax.jit(refine.refine, static_argnames=("kernel", ))
_ = ref(cv, exc, olf, ks).block_until_ready()

timeit(lambda: ref(cv, exc, olf, ks).block_until_ready())

# %%
olf, ks = refine.layer_refinement_matrices(distances0[:1], kernel)

cv = coarse_values.ravel().copy()
conv = jax.jit(
    partial(
        jax.lax.conv_general_dilated,
        window_strides=(1, ),
        padding="valid",
        dimension_numbers=("NWC", "OIW", "NWC")
    )
)
_ = conv(
    cv[:cv.shape[0] - cv.shape[0] % 3].reshape(1, -1, 3), olf.reshape(2, 3, 1)
)

timeit(
    lambda: conv(
        cv[:cv.shape[0] - cv.shape[0] % 3].reshape(1, -1, 3),
        olf.reshape(2, 3, 1)
    ).block_until_ready()
)
