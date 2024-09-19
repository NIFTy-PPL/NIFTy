#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import pytest

pytest.importorskip("jax")

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from numpy.testing import assert_allclose, assert_array_equal

import nifty8.re as jft

jax.config.update("jax_enable_x64", True)
pmp = pytest.mark.parametrize


def wiener_filter_posterior_dense(data, response, noise_covariance, prior_covariance):
    r"""Compute the posterior mean for the
    [Wiener filter](https://en.wikipedia.org/wiki/Information_field_theory#Generalized_Wiener_filter)
    setting.

    The maximally plausible $\rho_m$ is
    \begin{equation}
        \rho_m = (R^T N^{-1}R + S^{-1})^{-1} R^T N^{-1} \mathcal{D}_A
    \end{equation}
    with $N$ a matrix with $\sigma_{A,i}$ on its diagonal, $\blacksquare^T$ the
    [matrix-transpose](https://en.wikipedia.org/wiki/Transpose), and
    $\blacksquare^{-1}$ the
    [matrix-inverse](https://en.wikipedia.org/wiki/Invertible_matrix).
    """
    noise_covariance = (
        np.diag(noise_covariance) if noise_covariance.ndim == 1 else noise_covariance
    )
    noise_cov_inv = np.linalg.inv(noise_covariance)
    post_cov_inv = response.T @ noise_cov_inv @ response + np.linalg.inv(
        prior_covariance
    )
    post_cov = np.linalg.inv(post_cov_inv)
    return post_cov @ response.T @ noise_cov_inv @ data, post_cov


def los_response(end, shape, extent=(0, 1.0), start=0.0):
    """Compute the line-of-sight response matrix for a given shape and extent."""
    if len(shape) > 1:
        raise ValueError("only 1D shapes are supported.")
    start, end = np.broadcast_arrays(start, end)
    if np.any(start > end):
        raise ValueError("start must be smaller than end.")

    # Convert to dimensionless pixel indices
    start = (start - extent[0]) / (extent[1] - extent[0]) * (shape[0] - 1)
    end = (end - extent[0]) / (extent[1] - extent[0]) * (shape[0] - 1)
    response = []
    for i in range(shape[0]):
        # print(i, start, (i + 0.5 - start).clip(0., 1.))
        # Assume the zeroth pixel is centered at 0.0
        l = np.minimum((i + 0.5 - start).clip(0.0, 1.0), (end - i + 0.5).clip(0.0, 1.0))
        response.append(l)
    return np.stack(response, axis=1)


def signal_cov(shape, extent=(0, 1.0), scale=1.0, lengthscale=0.1):
    from scipy.spatial import distance_matrix

    if len(shape) > 1:
        raise ValueError("only 1D shapes are supported.")
    p = np.linspace(*extent, num=shape[0], endpoint=True)
    d = distance_matrix(p.reshape(-1, 1), p.reshape(-1, 1))
    return scale * (np.exp(-((d / lengthscale) ** 2)) + 1e-6 * np.eye(shape[0]))


@pmp("seed", (12, 42))
def test_mgvi_wiener_filter_consistency(
    seed,
    shape=(16,),
    extent=(0.0, 1.0),
    n_los=14,
    n_mgvi_cov_samples=16_000,
    n_vi_samples=16,
):
    key = random.PRNGKey(seed)
    size = np.prod(shape)

    key, k_r, k_p, k_n = random.split(key, 4)
    prior_cov = signal_cov(shape, extent=extent)
    prior_lsm = np.linalg.cholesky(prior_cov)
    dust_synth = prior_lsm @ random.normal(k_p, shape=shape)
    dust_synth = np.exp(dust_synth)  # Fake positivity; THIS VOILATES THE PRIOR
    endpoints = random.uniform(k_r, minval=extent[0], maxval=extent[1], shape=(n_los,))
    response = los_response(end=endpoints, shape=shape, extent=extent)
    k_n1, k_n2 = random.split(k_n)
    noise_cov = np.mean(dust_synth) * (
        1 + random.uniform(k_n1, minval=0.1, maxval=0.5, shape=(n_los,))
    )
    data = response @ dust_synth
    data += np.sqrt(noise_cov) * random.normal(k_n2, shape=(n_los,))
    # plt.scatter(endpoints, data); plt.show()

    # Retrieve Wiener Filter posterior (even though model draws data slightly
    # differently)
    post_mean, post_cov = wiener_filter_posterior_dense(
        data, response, noise_cov, prior_cov
    )
    # plt.plot(np.array((dust_synth, post_mean)).T); plt.show()
    # plt.matshow(post_cov); plt.colorbar(); plt.show()

    # Retrieve the equivalent Wiener filter solution for a standardized prior
    post_mean_st, post_cov_st = wiener_filter_posterior_dense(
        data, response @ prior_lsm, noise_cov, np.eye(size)
    )
    post_mean_alt = prior_lsm @ post_mean_st
    post_cov_alt = prior_lsm @ post_cov_st @ prior_lsm.T
    # Check for consistency with the "normal" Wiener filter solution
    assert_allclose(post_mean_alt, post_mean, atol=1e-8, rtol=1e-8)
    assert_allclose(post_cov_alt, post_cov, atol=1e-8, rtol=1e-8)

    # Retrieve posterior via NIFTy
    lh = jft.Gaussian(
        data, noise_cov_inv=partial(jnp.multiply, np.reciprocal(noise_cov))
    )
    forward = partial(jnp.matmul, prior_lsm)
    lh = lh.amend(partial(jnp.matmul, response)).amend(forward)
    lh._domain = jax.ShapeDtypeStruct(shape, dtype=jnp.float64)
    draw_linear_kwargs = dict(cg_kwargs=dict(absdelta=1e-10, maxiter=size))
    draw_linear_residual = partial(jft.draw_linear_residual, lh, **draw_linear_kwargs)
    draw_linear_residual = jft.smap(draw_linear_residual, in_axes=(None, 0))
    # Compare metric of Likelihood + Prior against true posterior covariance
    key, sk = random.split(key)
    some_positions = (random.normal(sk, shape=shape), jnp.ones(shape), jnp.zeros(shape))
    key, k_smpl = random.split(key)
    for pos in some_positions:
        probe = jnp.zeros(shape)
        lh_metric = jax.vmap(
            lambda i: lh.metric(pos, probe.at[i].set(1.0)), out_axes=1
        )(jnp.arange(size))
        lh_metric_diy = (
            prior_lsm.T @ response.T @ np.diag(1.0 / noise_cov) @ response @ prior_lsm
        )
        assert_allclose(lh_metric, lh_metric_diy, atol=1e-10, rtol=1e-12)
        mgvi_post_cov = jnp.linalg.inv(lh_metric + jnp.eye(size))
        # Compare posterior cov in space of standardized prior
        assert_allclose(mgvi_post_cov, post_cov_st, atol=1e-12, rtol=1e-14)
        # plt.matshow(mgvi_post_cov); plt.colorbar(); plt.show()

        # Test sampling approach
        residual_draw, info = draw_linear_residual(
            pos, random.split(k_smpl, n_mgvi_cov_samples)
        )
        assert_array_equal(info, 0)
        residual_draw = jax.vmap(forward)(residual_draw)
        tol = (np.diag(post_cov) / n_mgvi_cov_samples).sum()
        assert_allclose(residual_draw.mean(axis=0) ** 2, 0.0, atol=tol)
        smpl_post_cov = np.cov(residual_draw, rowvar=False)
        assert_allclose(smpl_post_cov, post_cov, atol=tol**0.5)

    kl_kwargs = dict(minimize_kwargs=dict(name="M", xtol=1e-10, maxiter=35))

    # MGVI test
    key, sk = random.split(key)
    samples_opt, _ = jft.optimize_kl(
        lh,
        pos,
        key=sk,
        n_total_iterations=1,
        n_samples=n_vi_samples,
        draw_linear_kwargs=draw_linear_kwargs,
        kl_kwargs=kl_kwargs,
        sample_mode="linear_sample",
        odir=None,
    )
    approx_post_mean = jax.vmap(forward)(samples_opt.samples).mean(axis=0)
    assert_allclose(post_mean, approx_post_mean, atol=1e-9, rtol=1e-10)

    delta = 1e-3
    nonlinearly_update_kwargs = dict(
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None, miniter=2),
            maxiter=5,
        )
    )

    # geoVI test
    key, sk = random.split(key)
    samples_opt, _ = jft.optimize_kl(
        lh,
        pos,
        key=sk,
        n_total_iterations=1,
        n_samples=n_vi_samples,
        draw_linear_kwargs=draw_linear_kwargs,
        nonlinearly_update_kwargs=nonlinearly_update_kwargs,
        kl_kwargs=kl_kwargs,
        sample_mode="nonlinear_sample",
        odir=None,
    )
    approx_post_mean = jax.vmap(forward)(samples_opt.samples).mean(axis=0)
    assert_allclose(post_mean, approx_post_mean, atol=1e-10, rtol=1e-10)

    # Wiener filter test
    key, w_key = random.split(key)
    wiener_samples, _ = jft.wiener_filter_posterior(
        lh,
        key=w_key,
        n_samples=n_mgvi_cov_samples,
        draw_linear_kwargs=draw_linear_kwargs,
    )
    wiener_post_mean = jax.vmap(forward)(wiener_samples.samples).mean(axis=0)
    wiener_post_cov = jnp.cov(jax.vmap(forward)(wiener_samples.samples), rowvar=False)
    assert_allclose(post_mean, wiener_post_mean, atol=7e-7, rtol=7e-7)
    assert_allclose(post_cov, wiener_post_cov, atol=tol**0.5)


if __name__ == "__main__":
    test_mgvi_wiener_filter_consistency(
        42,
        shape=(16,),
        extent=(0.0, 1.0),
        n_los=14,
        n_mgvi_cov_samples=16_000,
        n_vi_samples=16,
    )
