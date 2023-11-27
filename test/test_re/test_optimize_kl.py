#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import pytest

pytest.importorskip("jax")

from functools import partial, reduce

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from numpy.testing import assert_allclose, assert_array_equal

import nifty8.re as jft
from nifty8.re.optimize_kl import concatenate_zip

pmp = pytest.mark.parametrize


def random_draw(key, shape, dtype, method):
    def _isleaf(x):
        if isinstance(x, tuple):
            return reduce(lambda a, b: a * b, (isinstance(ii, int) for ii in x))
        return False

    swd = jax.tree_map(
        lambda x: jft.ShapeWithDtype(x, dtype), shape, is_leaf=_isleaf
    )
    return jft.random_like(key, jft.Vector(swd), method)


def random_noise_std_inv(key, shape):
    diag = 1. / random_draw(key, shape, float, random.exponential)

    def noise_std_inv(tangents):
        return diag * tangents

    return noise_std_inv


lh_init = (
    (
        jft.Gaussian,
        {
            "data": partial(random_draw, dtype=float, method=random.normal),
            "noise_std_inv": random_noise_std_inv
        },
        partial(random_draw, dtype=float, method=random.normal),
    ), (
        jft.StudentT,
        {
            "data": partial(random_draw, dtype=float, method=random.normal),
            "dof": partial(random_draw, dtype=float, method=random.exponential),
            "noise_std_inv": random_noise_std_inv
        },
        partial(random_draw, dtype=float, method=random.normal),
    ), (
        jft.Poissonian,
        {
            "data":
                partial(
                    random_draw,
                    dtype=int,
                    method=partial(random.poisson, lam=3.14)
                ),
        },
        lambda key, shape: jax.tree_map(
            lambda x: x.astype(float), 6. +
            random_draw(key, shape, int, partial(random.poisson, lam=3.14))
        ),
    ), (
        jft.VariableCovarianceGaussian,
        {
            "data": partial(random_draw, dtype=float, method=random.normal),
        },
        lambda key, shape: jft.Vector(
            (
                random_draw(key, shape, float, random.normal), 3. + 1. / jax.
                tree_map(
                    jnp.exp, random_draw(key, shape, float, random.normal)
                )
            )
        ),
    ), (
        jft.VariableCovarianceStudentT,
        {
            "data": partial(random_draw, dtype=float, method=random.normal),
            "dof": partial(random_draw, dtype=float, method=random.exponential),
        },
        lambda key, shape: jft.Vector(
            (
                random_draw(key, shape, float, random.normal),
                jax.tree_map(
                    jnp.exp, 3. + 1e-1 *
                    random_draw(key, shape, float, random.normal)
                )
            )
        ),
    )
)


@pmp("seed", (42, 43))
@pmp("shape", ((5, 12), (5, ), (1, 2, 3, 4)))
@pmp("ndim", (1, 2, 3))
def test_concatenate_zip(seed, shape, ndim):
    keys = random.split(random.PRNGKey(seed), ndim)
    zip_args = tuple(random.normal(k, shape=shape) for k in keys)
    assert_array_equal(
        concatenate_zip(*zip_args),
        np.concatenate(list(zip(*(list(el) for el in zip_args))))
    )


@pmp("seed", (12, 42))
@pmp("shape", ((4, 2), (2, 1), (5, ), [(2, 3), (1, 2)], ((2, ), {'a': (3, 1)})))
@pmp("lh_init", lh_init)
@pmp("sample_mode", ("linear", "nonlinear"))
def test_optimize_kl_sample_consistency(seed, shape, lh_init, sample_mode):
    rtol = 1e-7
    atol = 0.0
    aallclose = partial(assert_allclose, rtol=rtol, atol=atol)

    lh_init_method, draw, latent_init = lh_init
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, 1 + len(draw))
    init_kwargs = {
        k: method(key=sk, shape=shape)
        for (k, method), sk in zip(draw.items(), subkeys)
    }
    lh = lh_init_method(**init_kwargs)
    key, sk = random.split(key)
    pos = latent_init(sk, shape=shape)
    jft.tree_math.assert_arithmetics(pos)

    if lh._transformation is None and sample_mode == "nonlinear":
        pytest.skip("no transformation rule implemented yet")

    key, sk = random.split(key)
    delta = 1e-3
    absdelta = delta * jft.size(pos)

    draw_linear_samples = dict(
        cg_name="SL",
        cg_kwargs=dict(miniter=2, absdelta=absdelta / 10., maxiter=100),
    )
    minimize_kwargs = dict(
        name="SN",
        xtol=delta,
        cg_kwargs=dict(name=None, miniter=2),
        maxiter=5 if sample_mode.lower() == "nonlinear" else 0,
    )

    residual_draw, _ = jft.draw_residual(
        lh, pos, sk, **draw_linear_samples, minimize_kwargs=minimize_kwargs
    )
    residual_diy_l1, _ = jft.draw_linear_residual(
        lh, pos, sk, **draw_linear_samples
    )
    residual_diy = jft.stack((residual_diy_l1, -residual_diy_l1))
    if sample_mode.lower() == "linear":
        jax.tree_map(aallclose, residual_draw, residual_diy)

    residual_diy_n1, _ = jft.nonlinearly_update_residual(
        lh,
        pos,
        residual_diy_l1,
        metric_sample_key=sk,
        metric_sample_sign=+1,
        minimize_kwargs=minimize_kwargs,
        jit=False,
    )
    residual_diy_n2, _ = jft.nonlinearly_update_residual(
        lh,
        pos,
        -residual_diy_l1,
        metric_sample_key=sk,
        metric_sample_sign=-1,
        minimize_kwargs=minimize_kwargs,
        jit=False,
    )
    residual_diy = jft.stack((residual_diy_n1, residual_diy_n2))
    jax.tree_map(aallclose, residual_draw, residual_diy)

    samples_opt, _ = jft.optimize_kl(
        lh,
        pos,
        n_total_iterations=1,
        n_samples=1,
        key=sk,
        draw_linear_samples=draw_linear_samples,
        nonlinearly_update_samples=dict(minimize_kwargs=minimize_kwargs),
        minimize_kwargs=dict(name="M", maxiter=0),
        sample_mode=sample_mode,
        kl_jit=False,
        residual_jit=False,
        odir=None,
    )
    # effective key for sampling after splitting in `optimize_kl`
    _, ek = random.split(sk)
    residual_draw, _ = jft.draw_residual(
        lh, pos, ek, **draw_linear_samples, minimize_kwargs=minimize_kwargs
    )
    jax.tree_map(aallclose, samples_opt._samples, residual_draw)


if __name__ == "__main__":
    test_concatenate_zip(1, (5, 12), 3)
    test_optimize_kl_sample_consistency(1, (5, 4), lh_init[0], "linear")
    test_optimize_kl_sample_consistency(42, (5, 4), lh_init[4], "linear")
    test_optimize_kl_sample_consistency(
        42,
        ((2, ), {
            'a': (3, 1)
        }),
        lh_init[3],
        "nonlinear",
    )
