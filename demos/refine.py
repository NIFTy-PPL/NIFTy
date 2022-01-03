#!/usr/bin/env python3

from collections import namedtuple
from functools import partial
import sys

import jax
from jax import numpy as jnp
from jax import random
from jax import config as jax_config
from jax.scipy.special import gammaln
import matplotlib.pyplot as plt
import numpy as np

import jifty1 as jft

jax_config.update("jax_enable_x64", True)
interactive = False

Timed = namedtuple("Timed", ("time", "number"), rename=True)


def timeit(stmt, setup=lambda: None, number=None):
    import timeit

    if number is None:
        number, _ = timeit.Timer(stmt).autorange()

    setup()
    t = timeit.timeit(stmt, number=number) / number
    return Timed(time=t, number=number)


def matern_kernel(distance, scale, cutoff, dof):
    from scipy.special import kv
    reg_dist = jnp.sqrt(2 * dof) * distance / cutoff
    return scale**2 * 2**(1 - dof) / jnp.exp(
        gammaln(dof)
    ) * (reg_dist)**dof * kv(dof, reg_dist)


scale, cutoff, dof = 1., 80., 3 / 2

# %%
x = np.logspace(-6, 11, base=jnp.e, num=int(1e+5))
y = matern_kernel(x, scale, cutoff, dof)
y = jnp.nan_to_num(y, nan=0.)
kernel = partial(jnp.interp, xp=x, fp=y)
inv_kernel = partial(jnp.interp, xp=y, fp=x)

# fig, ax = plt.subplots()
# x_s = x[x < 10 * cutoff]
# ax.plot(x_s, kernel(x_s))
# ax.plot(x_s, jnp.exp(-(x_s / (2. * cutoff))**2))
# ax.set_yscale("log")
# plt.show()

# %%
coord = jnp.linspace(0., 500., num=500)
coord = coord.reshape(1, -1)

kern_outer = jax.vmap(
    jax.vmap(lambda x, y: kernel(jnp.linalg.norm(x - y)), in_axes=(None, 1)),
    in_axes=(1, None)
)
# plt.imshow(kern_outer(coord, coord))
# plt.colorbar()
# plt.show()

# %%
key = random.PRNGKey(42)
key, k_c, k_f = random.split(key, 3)

main_coord = coord[:, ::10]
cov_sqrt = jnp.linalg.cholesky(kern_outer(main_coord, main_coord))
lvl0 = cov_sqrt @ random.normal(k_c, shape=main_coord.shape[::-1])

coarse_coord = main_coord[:, :3]
fine_coord = coarse_coord[tuple(jnp.array(coarse_coord.shape) // 2)
                         ] + (jnp.diff(coarse_coord) / jnp.array([-4., 4.]))
cov_ff = kern_outer(fine_coord, fine_coord)
cov_fc = kern_outer(fine_coord, coarse_coord)
cov_cc_inv = jnp.linalg.inv(kern_outer(coarse_coord, coarse_coord))

fine_kernel = cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
fine_kernel_sqrt = jnp.linalg.cholesky(fine_kernel)
olf = cov_fc @ cov_cc_inv

# %%
# N - batch dimension
# H - spatial height
# W - spatial width
# D - spatial depth
# C - channel dimension
# I - kernel input channel dimension
# O - kernel output channel dimension
#dim_nums = ('NCWH', 'IWHO', 'NWHC')

refined_m = jax.vmap(
    partial(jnp.convolve, mode="valid"), in_axes=(None, 0), out_axes=1
)(lvl0.ravel(), olf[::-1])

lvl1_exc = random.normal(k_f, shape=refined_m.shape)
refined_std = jax.vmap(jnp.matmul,
                       in_axes=(None, 0))(fine_kernel_sqrt, lvl1_exc)
lvl1 = (refined_m + refined_std).ravel()

lvl1_full_coord = main_coord[
    ..., np.newaxis] + jnp.diff(coarse_coord) / jnp.array([-4., 4.])
lvl1_full_coord = lvl1_full_coord[:, 1:-1, :].ravel()
if interactive:
    plt.plot(main_coord.ravel(), lvl0)
    plt.plot(lvl1_full_coord, lvl1)
    plt.show()


# %%
def layer_refinement_matrices(distances, kernel):
    def cov_from_loc_sngl(x, y):
        return kernel(jnp.linalg.norm(x - y))

    # TODO: more dimensions
    cov_from_loc = jax.vmap(
        jax.vmap(cov_from_loc_sngl, in_axes=(None, 0)), in_axes=(0, None)
    )

    coarse_coord = distances * jnp.array([-1., 0., 1.])
    fine_coord = distances * jnp.array([-.25, .25])
    cov_ff = cov_from_loc(fine_coord, fine_coord)
    cov_fc = cov_from_loc(fine_coord, coarse_coord)
    cov_cc_inv = jnp.linalg.inv(cov_from_loc(coarse_coord, coarse_coord))

    olf = cov_fc @ cov_cc_inv
    fine_kernel_sqrt = jnp.linalg.cholesky(
        cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
    )

    return olf, fine_kernel_sqrt


def refinement_matrices_alt(size0, depth, distances, kernel):
    coord0 = distances * jnp.arange(size0, dtype=float)
    cov_sqrt0 = jnp.linalg.cholesky(cov_from_loc(coord0, coord0))

    dist_by_depth = distances * 0.5**jnp.arange(1, depth)
    opt_lin_filter, kernel_sqrt = jax.vmap(
        partial(layer_refinement_matrices, kernel=kernel),
        in_axes=0,
        out_axes=(0, 0)
    )(dist_by_depth)

    return opt_lin_filter, (cov_sqrt0, kernel_sqrt)


def refinement_matrices(size0, depth, distances, kernel):
    #  Roughly twice as faster compared to vmapped `layer_refinement_matrices`

    def cov_from_loc(x, y):
        mat = jnp.subtract(*jnp.meshgrid(x, y, indexing="ij"))
        return kernel(jnp.linalg.norm(mat[..., jnp.newaxis], axis=-1))

    def olaf(dist):
        coord = dist * jnp.array([-1., 0., 1., -0.25, 0.25])
        cov = cov_from_loc(coord, coord)
        cov_ff = cov[-2:, -2:]
        cov_fc = cov[-2:, :-2]
        cov_cc_inv = jnp.linalg.inv(cov[:-2, :-2])

        olf = cov_fc @ cov_cc_inv
        fine_kernel_sqrt = jnp.linalg.cholesky(
            cov_ff - cov_fc @ cov_cc_inv @ cov_fc.T
        )

        return olf, fine_kernel_sqrt

    coord0 = distances * jnp.arange(size0, dtype=float)
    cov_sqrt0 = jnp.linalg.cholesky(cov_from_loc(coord0, coord0))

    dist_by_depth = distances * 0.5**jnp.arange(1, depth)
    opt_lin_filter, kernel_sqrt = jax.vmap(olaf, in_axes=0,
                                           out_axes=(0, 0))(dist_by_depth)

    return opt_lin_filter, (cov_sqrt0, kernel_sqrt)


def refine_conv(coarse_values, excitations, olf, fine_kernel_sqrt):
    fine_m = jax.vmap(
        partial(jnp.convolve, mode="valid"), in_axes=(None, 0), out_axes=0
    )(coarse_values, olf[::-1])
    fine_m = jnp.moveaxis(fine_m, (0, ), (1, ))
    fine_std = jax.vmap(jnp.matmul, in_axes=(None, 0))(
        fine_kernel_sqrt, excitations.reshape(-1, fine_kernel_sqrt.shape[-1])
    )

    return (fine_m + fine_std).ravel()


def refine_loop(coarse_values, excitations, olf, fine_kernel_sqrt):
    fine_m = [jnp.convolve(coarse_values, o, mode="valid") for o in olf[::-1]]
    fine_m = jnp.stack(fine_m, axis=1)
    fine_std = jax.vmap(jnp.matmul, in_axes=(None, 0))(
        fine_kernel_sqrt, excitations.reshape(-1, fine_kernel_sqrt.shape[-1])
    )

    return (fine_m + fine_std).ravel()


def refine_conv_general(coarse_values, excitations, olf, fine_kernel_sqrt):
    olf = olf[..., jnp.newaxis]

    sh0 = coarse_values.shape[0]
    conv = partial(
        jax.lax.conv_general_dilated,
        window_strides=(1, ),
        padding="valid",
        dimension_numbers=("NWC", "OIW", "NWC")
    )
    fine_m = jnp.zeros((coarse_values.size - 2, 2))
    fine_m = fine_m.at[0::3].set(
        conv(coarse_values[:sh0 - sh0 % 3].reshape(1, -1, 3), olf)[0]
    )
    fine_m = fine_m.at[1::3].set(
        conv(coarse_values[1:sh0 - (sh0 - 1) % 3].reshape(1, -1, 3), olf)[0]
    )
    fine_m = fine_m.at[2::3].set(
        conv(coarse_values[2:sh0 - (sh0 - 2) % 3].reshape(1, -1, 3), olf)[0]
    )

    fine_std = jax.vmap(jnp.matmul, in_axes=(None, 0))(
        fine_kernel_sqrt, excitations.reshape(-1, fine_kernel_sqrt.shape[-1])
    )

    return (fine_m + fine_std).ravel()


def refine_vmap(coarse_values, excitations, olf, fine_kernel_sqrt):
    sh0 = coarse_values.shape[0]
    conv = jax.vmap(jnp.matmul, in_axes=(None, 0), out_axes=0)
    fine_m = jnp.zeros((coarse_values.size - 2, 2))
    fine_m = fine_m.at[0::3].set(
        conv(olf, coarse_values[:sh0 - sh0 % 3].reshape(-1, 3))
    )
    fine_m = fine_m.at[1::3].set(
        conv(olf, coarse_values[1:sh0 - (sh0 - 1) % 3].reshape(-1, 3))
    )
    fine_m = fine_m.at[2::3].set(
        conv(olf, coarse_values[2:sh0 - (sh0 - 2) % 3].reshape(-1, 3))
    )

    fine_std = jax.vmap(jnp.matmul, in_axes=(None, 0))(
        fine_kernel_sqrt, excitations.reshape(-1, fine_kernel_sqrt.shape[-1])
    )

    return (fine_m + fine_std).ravel()


refine = refine_conv

distances0 = 10.
distances1 = distances0 / 2
distances2 = distances1 / 2

key, k_c, *k_f = random.split(key, 6)

main_coord = jnp.linspace(0., 1000., 50)
cov_from_loc = jax.vmap(
    jax.vmap(lambda x, y: kernel(jnp.linalg.norm(x - y)), in_axes=(None, 0)),
    in_axes=(0, None)
)
cov_sqrt = jnp.linalg.cholesky(cov_from_loc(main_coord, main_coord))
lvl0 = cov_sqrt @ random.normal(k_c, shape=main_coord.shape[::-1])

lvl1_exc = random.normal(k_f[1], shape=(2 * (lvl0.size - 2), ))
lvl1 = refine(
    lvl0.ravel(), lvl1_exc, *layer_refinement_matrices(distances0, kernel)
)
lvl2_exc = random.normal(k_f[2], shape=(2 * (lvl1.size - 2), ))
lvl2 = refine(
    lvl1.ravel(), lvl2_exc, *layer_refinement_matrices(distances1, kernel)
)
lvl3_exc = random.normal(k_f[3], shape=(2 * (lvl2.size - 2), ))
lvl3 = refine(
    lvl2.ravel(), lvl3_exc, *layer_refinement_matrices(distances2, kernel)
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
    plt.step(x0, lvl0.ravel(), alpha=0.7, where="mid", label="LVL0")
    plt.step(x1, lvl1, alpha=0.7, where="mid", label="LVL1")
    # plt.step(x2, lvl2, alpha=0.7, where="mid", label="LVL2")
    # plt.step(x3, lvl3, alpha=0.7, where="mid", label="LVL3")
    plt.legend()
    plt.show()


# %%
def fwd(xi, distances, kernel):
    size0, depth = xi[0].size, len(xi)
    os, (cov_sqrt0, ks) = refinement_matrices(
        size0, depth, distances=distances, kernel=kernel
    )

    fine = cov_sqrt0 @ xi[0]
    for x, olf, ks in zip(xi[1:], os, ks):
        fine = refine(fine, x, olf, ks)
    return fine


key = random.PRNGKey(44)
key, *ks = random.split(key, 4)

distances = 50. * 1000
n_std = 0.1

n0_pix = 12
n_layers = 15
exc_shp = [(n0_pix, )]
for _ in range(n_layers):
    exc_shp.append((2 * (exc_shp[-1][0] - 2), ))

xi_truth_swd = list(map(jft.ShapeWithDtype, exc_shp))
xi_truth = jft.random_like(ks.pop(), xi_truth_swd)
d = fwd(xi_truth, distances, kernel)
d += n_std * random.normal(ks.pop(), shape=d.shape)


def signal_response(xi, distances):
    def kernel(x, cutoff):
        cutoff = jnp.exp(0.05 * cutoff + 1.5)
        return jnp.exp(-(x / cutoff)**2)

    return fwd(
        xi["excitations"], distances, partial(kernel, cutoff=xi["cutoff"])
    )


xi_swd = {"excitations": xi_truth_swd, "cutoff": jft.ShapeWithDtype(())}
xi = 1e-3 * jft.Field(jft.random_like(ks.pop(), xi_swd))
ham = lambda x: jnp.linalg.norm(d - signal_response(x, distances), ord=2) / (
    2 * n_std**2
) + 0.5 * jft.norm(x, ord=2, ravel=True)
ham = jax.jit(ham)

if interactive:
    plt.plot(d, label="Data")
    plt.plot(fwd(xi_truth, distances, kernel), label="Truth")
    plt.plot(signal_response(xi, distances), label="Start")
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

# %%
if interactive:
    plt.plot(d, label="Data")
    plt.plot(fwd(xi_truth, distances, kernel), label="Truth")
    plt.plot(signal_response(xi, distances), label="Start")
    plt.plot(signal_response(opt_state.x, distances), label="Final")
    plt.legend()
    plt.show()

# %%
# JIFTy CF :: Shape        (262144,) (    10 loops) :: JAX 2.83e-02 :: NIFTy 3.93e-02
# # # GRID REFINEMENT (size0 = 12) CPU: 8 cores 22 GB; GPU: A100 40 GB (refine_conv_general)
# GRID REFINEMENT :: CPU :: Shape        (262148,) (    20 loops) :: JAX w/ learnable 1.13e-02
# GRID REFINEMENT :: CPU :: Shape        (262148,) (    50 loops) :: JAX w/o learnable 8.91e-03
# GRID REFINEMENT :: GPU :: Shape        (262148,) (    50 loops) :: JAX w/ learnable 9.62e-03
# GRID REFINEMENT :: GPU :: Shape        (262148,) (   100 loops) :: JAX w/o learnable 2.78e-03
# # # GRID REFINEMENT (size0 = 12) CPU: 8 cores 22 GB; GPU: A100 40 GB (refine_vmap)
# GRID REFINEMENT :: CPU :: Shape        (262148,) (    20 loops) :: JAX w/ learnable 1.09e-02
# GRID REFINEMENT :: CPU :: Shape        (262148,) (    50 loops) :: JAX w/o learnable 8.08e-03
# GRID REFINEMENT :: GPU :: Shape        (262148,) (    50 loops) :: JAX w/ learnable 8.95e-03
# GRID REFINEMENT :: GPU :: Shape        (262148,) (   100 loops) :: JAX w/o learnable 2.21e-03
# # # GRID REFINEMENT (size0 = 12) CPU: 8 cores 22 GB; GPU: A100 40 GB (refine_conv)
# GRID REFINEMENT :: CPU :: Shape        (262148,) (    50 loops) :: JAX w/ learnable 5.80e-03
# GRID REFINEMENT :: CPU :: Shape        (262148,) (   100 loops) :: JAX w/o learnable 3.80e-03
# GRID REFINEMENT :: GPU :: Shape        (262148,) (    10 loops) :: JAX w/ learnable 2.34e-02
# GRID REFINEMENT :: GPU :: Shape        (262148,) (   200 loops) :: JAX w/o learnable 1.44e-03
# # # GRID REFINEMENT (size0 = 12) CPU: 8 cores 22 GB; GPU: A100 40 GB (refine_loop)
# GRID REFINEMENT :: CPU :: Shape        (262148,) (    50 loops) :: JAX w/ learnable 6.19e-03
# GRID REFINEMENT :: CPU :: Shape        (262148,) (    50 loops) :: JAX w/o learnable 4.55e-03
# GRID REFINEMENT :: GPU :: Shape        (262148,) (    50 loops) :: JAX w/ learnable 8.23e-03
# GRID REFINEMENT :: GPU :: Shape        (262148,) (   200 loops) :: JAX w/o learnable 1.66e-03
for backend in ("cpu", "gpu"):
    device_kw = {"device": jax.devices(backend=backend)[0]}
    device_put = partial(jax.device_put, **device_kw)

    corr = jax.jit(
        jax.value_and_grad(
            lambda x: ((1. - signal_response(x, distances))**2).sum()
        ), **device_kw
    )
    x = xi

    x = device_put(x)
    _ = corr(x)[0].block_until_ready()
    t = timeit(lambda: corr(x)[0].block_until_ready())
    ti, num = t.time, t.number

    print(
        f"{backend.upper()} :: Shape {str(exc_shp[-1]):>16s} ({num:6d} loops) :: JAX w/ learnable {ti:4.2e}",
        file=sys.stderr
    )

    corr = jax.jit(
        jax.
        value_and_grad(lambda x: ((1. - fwd(x, distances, kernel))**2).sum()),
        **device_kw
    )
    x = xi_truth

    x = device_put(x)
    _ = corr(x)[0].block_until_ready()
    t = timeit(lambda: corr(x)[0].block_until_ready())
    ti, num = t.time, t.number

    print(
        f"{backend.upper()} :: Shape {str(exc_shp[-1]):>16s} ({num:6d} loops) :: JAX w/o learnable {ti:4.2e}",
        file=sys.stderr
    )

# %%
coarse_values = fwd(xi_truth, distances, kernel)
olf, fine_kernel_sqrt = layer_refinement_matrices(distances, kernel)

cv = coarse_values
exc = random.normal(key, shape=(2 * (cv.size - 2), ))
ref = jax.jit(refine)
_ = ref(cv, exc, olf, fine_kernel_sqrt)
timeit(lambda: ref(cv, exc, olf, fine_kernel_sqrt).block_until_ready())


# %%
def fwd_diy(xi, opt_lin_filter, kernel_sqrt):
    cov_sqrt0, ks = kernel_sqrt
    fine = cov_sqrt0 @ xi[0]
    for x, olf, ks in zip(xi[1:], opt_lin_filter, ks):
        fine = refine(fine, x, olf, ks)
    return fine


x = jft.random_like(key, jft.Field(xi_truth_swd))
olfs, kss = refinement_matrices(x.val[0].size, len(x.val), distances, kernel)
corr = jax.jit(jax.value_and_grad(lambda x: fwd_diy(x, olfs, kss).sum()), )

_ = corr(x)[0].block_until_ready()
timeit(lambda: corr(x)[0].block_until_ready())

# %%
rm = jax.jit(layer_refinement_matrices, static_argnames=("kernel", ))
_ = rm(distances, kernel)
timeit(lambda: rm(distances, kernel)[0].block_until_ready())

# %%
grm = jax.jit(
    refinement_matrices, static_argnames=(
        "size0",
        "depth",
        "kernel",
    )
)
_ = grm(x.val[0].size, len(x.val), distances, kernel)
timeit(
    lambda: grm(x.val[0].size, len(x.val), distances, kernel)[0].
    block_until_ready()
)

# %%
cv = coarse_values.copy()
exc = random.normal(key, shape=(2 * (cv.size - 2), ))
olf, ks = layer_refinement_matrices(distances, kernel)
ref = jax.jit(refine, static_argnames=("kernel", ))
_ = ref(cv, exc, olf, ks).block_until_ready()

timeit(lambda: ref(cv, exc, olf, ks).block_until_ready())

# %%
cv = coarse_values.copy()
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
