#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import pytest

pytest.importorskip("jax")

from functools import partial, reduce

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import tree_map
from numpy.testing import assert_allclose, assert_array_equal

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
        jit=False,
    )
    residual_diy_n2, _ = jft.nonlinearly_update_residual(
        lh,
        pos,
        -residual_diy_l1,
        metric_sample_key=sk,
        metric_sample_sign=-1,
        point_estimates=point_estimates,
        minimize_kwargs=minimize_kwargs,
        jit=False,
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


if __name__ == "__main__":
    test_concatenate_zip(1, (5, 12), 3)
    test_optimize_kl_sample_consistency(
        1,
        ((2, [1, 1]), {"a": (3, 1)}),
        LH_INIT[1],
        random_point_estimates=True,
        sample_mode="nonlinear_resample",
    )
