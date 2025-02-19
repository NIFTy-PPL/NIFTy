#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import os

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=4"  # Use 4 CPU devices
)

import pytest

pytest.importorskip("jax")

from functools import partial, reduce

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import tree_map
from numpy.testing import assert_allclose, assert_array_equal

jax.config.update("jax_enable_x64", True)

import nifty8.re as jft
from nifty8.re.optimize_kl import concatenate_zip

pmp = pytest.mark.parametrize


def random_draw(key, shape, dtype, method):
    def _isleaf(x):
        if isinstance(x, tuple):
            return bool(reduce(lambda a, b: a * b, (isinstance(ii, int) for ii in x)))
        return False

    swd = tree_map(lambda x: jft.ShapeWithDtype(x, dtype), shape, is_leaf=_isleaf)
    return jft.random_like(key, jft.Vector(swd), method)


def random_noise_std_inv(key, shape):
    diag = 1.0 / random_draw(key, shape, float, random.exponential)

    def noise_std_inv(tangents):
        return diag * tangents

    return noise_std_inv


LH_INIT = (
    (
        jft.Gaussian,
        {
            "data": partial(random_draw, dtype=float, method=random.normal),
            "noise_std_inv": random_noise_std_inv,
        },
        partial(random_draw, dtype=float, method=random.normal),
    ),
    (
        jft.StudentT,
        {
            "data": partial(random_draw, dtype=float, method=random.normal),
            "dof": partial(random_draw, dtype=float, method=random.exponential),
            "noise_std_inv": random_noise_std_inv,
        },
        partial(random_draw, dtype=float, method=random.normal),
    ),
    (
        jft.Poissonian,
        {
            "data": partial(
                random_draw, dtype=int, method=partial(random.poisson, lam=3.14)
            ),
        },
        lambda key, shape: tree_map(
            lambda x: x.astype(float),
            6.0 + random_draw(key, shape, int, partial(random.poisson, lam=3.14)),
        ),
    ),
    (
        jft.VariableCovarianceGaussian,
        {
            "data": partial(random_draw, dtype=float, method=random.normal),
        },
        lambda key, shape: jft.Vector(
            (
                random_draw(key, shape, float, random.normal),
                3.0
                + 1.0
                / tree_map(jnp.exp, random_draw(key, shape, float, random.normal)),
            )
        ),
    ),
    (
        jft.VariableCovarianceStudentT,
        {
            "data": partial(random_draw, dtype=float, method=random.normal),
            "dof": partial(random_draw, dtype=float, method=random.exponential),
        },
        lambda key, shape: jft.Vector(
            (
                random_draw(key, shape, float, random.normal),
                tree_map(
                    jnp.exp, 3.0 + 1e-1 * random_draw(key, shape, float, random.normal)
                ),
            )
        ),
    ),
)


def _assert_zero_point_estimate_residuals(residuals, point_estimates):
    if not point_estimates:
        return
    assert_array_equal(jft.sum(point_estimates * residuals**2), 0.0)


@pmp("seed", (42, 43))
@pmp("shape", ((5, 12), (5,), (1, 2, 3, 4)))
@pmp("ndim", (1, 2, 3))
def test_concatenate_zip(seed, shape, ndim):
    keys = random.split(random.PRNGKey(seed), ndim)
    zip_args = tuple(random.normal(k, shape=shape) for k in keys)
    assert_array_equal(
        concatenate_zip(*zip_args),
        np.concatenate(list(zip(*(list(el) for el in zip_args)))),
    )


@pmp("seed", (12, 42))
@pmp("shape", ((5,), ((4,), (2,), (1,), (1,)), ((2, [1, 1]), {"a": (3, 1)})))
@pmp("lh_init", LH_INIT)
@pmp("random_point_estimates", (True, False))
@pmp("sample_mode", ("linear_resample", "nonlinear_resample"))
def test_optimize_kl_sample_consistency(
    seed, shape, lh_init, random_point_estimates, sample_mode
):
    rtol = 1e-7
    atol = 0.0
    aallclose = partial(assert_allclose, rtol=rtol, atol=atol)

    lh_init_method, draw, latent_init = lh_init
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, 1 + len(draw))
    init_kwargs = {
        k: method(key=sk, shape=shape) for (k, method), sk in zip(draw.items(), subkeys)
    }
    lh = lh_init_method(**init_kwargs)
    key, sk = random.split(key)
    pos = latent_init(sk, shape=shape)
    jft.tree_math.assert_arithmetics(pos)

    if sample_mode == "nonlinear_resample":
        try:
            lh.transformation(pos)
        except NotImplementedError:
            pytest.skip("no transformation rule implemented yet")

    key, sk = random.split(key)
    delta = 1e-3
    absdelta = delta * jft.size(pos)

    draw_linear_kwargs = dict(
        cg_name="SL",
        cg_kwargs=dict(miniter=2, absdelta=absdelta / 10.0, maxiter=100),
    )
    minimize_kwargs = dict(
        name="SN",
        xtol=delta,
        cg_kwargs=dict(name=None, miniter=2),
        maxiter=5 if sample_mode.lower() == "nonlinear_resample" else 0,
    )
    key, sk = random.split(key)
    point_estimates = ()
    if random_point_estimates:
        prob = 0.5
        pe, pe_td = jax.tree_util.tree_flatten(pos)
        pe = list(random.bernoulli(sk, prob, shape=(len(pe),)))
        pe[0] = False  # Ensure at least one variable is not a point-estimate
        pe = tuple(map(bool, pe))
        point_estimates = jax.tree_util.tree_unflatten(pe_td, pe)

    residual_draw, _ = jft.draw_residual(
        lh,
        pos,
        sk,
        point_estimates=point_estimates,
        minimize_kwargs=minimize_kwargs,
        **draw_linear_kwargs,
    )
    _assert_zero_point_estimate_residuals(residual_draw, point_estimates)
    residual_diy_l1, _ = jft.draw_linear_residual(
        lh, pos, sk, point_estimates=point_estimates, **draw_linear_kwargs
    )
    residual_diy = jft.stack((residual_diy_l1, -residual_diy_l1))
    _assert_zero_point_estimate_residuals(residual_diy, point_estimates)
    if sample_mode.lower() == "linear":
        tree_map(aallclose, residual_draw, residual_diy)

    residual_diy_n1, _ = jft.nonlinearly_update_residual(
        lh,
        pos,
        residual_diy_l1,
        metric_sample_key=sk,
        metric_sample_sign=+1,
        point_estimates=point_estimates,
        minimize_kwargs=minimize_kwargs,
    )
    residual_diy_n2, _ = jft.nonlinearly_update_residual(
        lh,
        pos,
        -residual_diy_l1,
        metric_sample_key=sk,
        metric_sample_sign=-1,
        point_estimates=point_estimates,
        minimize_kwargs=minimize_kwargs,
    )
    residual_diy = jft.stack((residual_diy_n1, residual_diy_n2))
    _assert_zero_point_estimate_residuals(residual_diy, point_estimates)
    tree_map(aallclose, residual_draw, residual_diy)

    key, sk = random.split(key)
    samples_opt, _ = jft.optimize_kl(
        lh,
        pos,
        key=sk,
        n_total_iterations=1,
        n_samples=1,
        point_estimates=point_estimates,
        draw_linear_kwargs=draw_linear_kwargs,
        nonlinearly_update_kwargs=dict(minimize_kwargs=minimize_kwargs),
        kl_kwargs=dict(minimize_kwargs=dict(name="M", maxiter=0)),
        sample_mode=sample_mode,
        kl_jit=False,
        residual_jit=False,
        odir=None,
    )
    _assert_zero_point_estimate_residuals(samples_opt._samples, point_estimates)
    residual_draw, _ = jft.draw_residual(
        lh,
        pos,
        samples_opt.keys[0],
        point_estimates=point_estimates,
        minimize_kwargs=minimize_kwargs,
        **draw_linear_kwargs,
    )
    _assert_zero_point_estimate_residuals(residual_draw, point_estimates)
    tree_map(aallclose, samples_opt._samples, residual_draw)


@pmp("seed", (12, 42))
@pmp("shape", ((5,), ((4,), (2,), (1,), (1,)), ((2, [1, 1]), {"a": (3, 1)})))
@pmp("lh_init", LH_INIT)
def test_optimize_kl_constants(seed, shape, lh_init):
    lh_init_method, draw, latent_init = lh_init
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, 1 + len(draw))
    init_kwargs = {
        k: method(key=sk, shape=shape) for (k, method), sk in zip(draw.items(), subkeys)
    }
    lh = lh_init_method(**init_kwargs)
    key, sk = random.split(key)
    pos = latent_init(sk, shape=shape)
    jft.tree_math.assert_arithmetics(pos)

    key, sk = random.split(key)
    delta = 1e-3
    absdelta = delta * jft.size(pos)

    draw_linear_kwargs = dict(
        cg_name="SL", cg_kwargs=dict(miniter=2, absdelta=absdelta / 10.0, maxiter=100)
    )
    minimize_kwargs = dict(
        name="SN", xtol=delta, cg_kwargs=dict(name=None, miniter=2), maxiter=5
    )

    # Assign random constants
    prob = 0.5
    const, const_td = jax.tree_util.tree_flatten(pos)
    const = list(random.bernoulli(sk, prob, shape=(len(const),)))
    const[0] = False  # Ensure at least one variable is not constant
    const = tuple(map(bool, const))
    constants = jax.tree_util.tree_unflatten(const_td, const)

    key, sk = random.split(key)
    samples_opt, _ = jft.optimize_kl(
        lh,
        pos,
        key=sk,
        n_total_iterations=1,
        n_samples=1,
        constants=constants,
        draw_linear_kwargs=draw_linear_kwargs,
        nonlinearly_update_kwargs=dict(minimize_kwargs=minimize_kwargs),
        kl_kwargs=dict(minimize_kwargs=dict(name="M", maxiter=5)),
        sample_mode="linear_resample",
        kl_jit=False,
        residual_jit=False,
        odir=None,
    )
    move, _ = jax.tree_util.tree_flatten(samples_opt.pos - pos)
    np.testing.assert_array_equal(
        0, np.concatenate([np.ravel(m) if c else [0] for m, c in zip(move, const)])
    )
    np.testing.assert_array_equal(
        True,
        0 != np.concatenate([[1] if c else np.ravel(m) for m, c in zip(move, const)]),
    )


@pmp("shape", (((2, [1, 1]), {"a": (3, 1)})))
@pmp(
    "sample_mode",
    (
        "linear_resample",
        "linear_sample",
        "nonlinear_resample",
        "nonlinear_sample",
        "nonlinear_update",
    ),
)
@pmp("n_samples", (2, 4, 8))
@pmp("residual_device_map", ("pmap", "shared_map"))
def test_optimize_kl_device_consistency(
    shape, sample_mode, n_samples, residual_device_map
):
    devices = jax.devices()
    if not len(devices) > 1:
        raise RuntimeError("Need more than one device for test.")
    if residual_device_map == "pmap" and n_samples > len(devices):
        pytest.skip("n_samples>len(devices), skipping for pmap.")
    lh_init_method, draw, latent_init = LH_INIT[0]
    key = random.PRNGKey(42)
    key, *subkeys = random.split(key, 1 + len(draw))
    init_kwargs = {
        k: method(key=sk, shape=shape) for (k, method), sk in zip(draw.items(), subkeys)
    }
    lh = lh_init_method(**init_kwargs)
    key, sk = random.split(key)
    pos = latent_init(sk, shape=shape)

    key, sk = random.split(key)
    delta = 1e-3
    absdelta = delta * jft.size(pos)

    draw_linear_kwargs = dict(
        cg_name="SL", cg_kwargs=dict(miniter=2, absdelta=absdelta / 10.0, maxiter=10)
    )
    minimize_kwargs = dict(
        name="SN", xtol=delta, cg_kwargs=dict(name=None, miniter=2), maxiter=5
    )

    key, sk = random.split(key)
    opt_kl_kwargs = dict(
        likelihood=lh,
        position_or_samples=pos,
        key=sk,
        n_total_iterations=2,
        n_samples=n_samples,
        draw_linear_kwargs=draw_linear_kwargs,
        nonlinearly_update_kwargs=dict(minimize_kwargs=minimize_kwargs),
        kl_kwargs=dict(minimize_kwargs=dict(name="M", maxiter=10)),
        sample_mode=sample_mode,
        odir=None,
        residual_map="smap",
    )
    samples_single_device, _ = jft.optimize_kl(**opt_kl_kwargs, map_over_devices=None)
    samples_multiple_devices, _ = jft.optimize_kl(
        **opt_kl_kwargs,
        map_over_devices=jax.devices(),
        residual_device_map=residual_device_map,
    )
    aallclose = partial(assert_allclose, rtol=1e-5, atol=1e-5)
    tree_map(
        aallclose,
        samples_single_device.samples,
        samples_multiple_devices.samples,
    )
    tree_map(aallclose, samples_single_device.pos, samples_multiple_devices.pos)
    tree_map(
        assert_array_equal, samples_single_device.keys, samples_multiple_devices.keys
    )


if __name__ == "__main__":
    test_concatenate_zip(1, (5, 12), 3)
    test_optimize_kl_sample_consistency(
        1,
        ((2, [1, 1]), {"a": (3, 1)}),
        LH_INIT[1],
        random_point_estimates=True,
        sample_mode="nonlinear_resample",
    )
    test_optimize_kl_constants(1, ((2, [1, 1]), {"a": (3, 1)}), LH_INIT[1])
