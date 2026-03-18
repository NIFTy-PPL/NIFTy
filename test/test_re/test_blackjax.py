#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import jax
import jax.numpy as jnp
from jax import random
from numpy.testing import assert_allclose
import nifty.re as jft

jax.config.update("jax_enable_x64", True)


def init_model(key, dims=30, distances=0.01, noise_std=0.1):
    grid = jft.correlated_field.make_grid(
        dims, distances=distances, harmonic_type="fourier"
    )

    k = grid.harmonic_grid.mode_lengths
    amp = 2.5 / (100 + k**2.5)
    sqrt_cov = amp[grid.harmonic_grid.power_distributor]

    class FixedPowerCF(jft.Model):
        def __init__(self):
            self.ht = jft.correlated_field.hartley
            self.sqrt_cov = sqrt_cov
            self.harmonic_dvol = 1 / grid.total_volume
            super().__init__(domain=jax.ShapeDtypeStruct(grid.shape, jnp.float64))

        def __call__(self, x):
            return self.harmonic_dvol * self.ht(self.sqrt_cov * x)

    signal = FixedPowerCF()

    key, k1, k2 = random.split(key, 3)
    pos_truth = jft.random_like(k1, signal.domain)
    data = signal(pos_truth) + noise_std * jft.random_like(k2, signal.target)

    lh = jft.Gaussian(data, lambda x: x / noise_std**2).amend(signal)
    return lh, signal


def test_nuts_against_geovi(seed=42):
    key = random.PRNGKey(seed)
    key, k_model, k_wf, k_init, k_nuts = random.split(key, 5)

    lh, signal = init_model(k_model)
    delta = 1e-6

    wf, _ = jft.wiener_filter_posterior(
        likelihood=lh,
        key=k_wf,
        n_samples=0,
        draw_linear_kwargs=dict(
            cg_name=None,
            cg_kwargs=dict(
                absdelta=delta * jft.size(lh.domain) / 10.0,
                maxiter=30,
            ),
        ),
    )

    pos_init = lh.init(k_init)
    samples, _ = jft.blackjax_nuts(
        lh,
        pos_init,
        k_nuts,
        n_warmup_steps=100,
        n_samples=100,
    )

    mean_wf = signal(wf.pos)
    mean_nuts, _ = jft.mean_and_std(tuple(signal(s) for s in samples))

    assert_allclose(mean_wf, mean_nuts, atol=2e-1, rtol=2e-1)


def test_nuts_resume(seed=42):
    key = random.PRNGKey(seed)
    key, k_model, k_init, k_nuts, k_resume = random.split(key, 5)

    lh, _ = init_model(k_model)
    pos_init = lh.init(k_init)

    _, state = jft.blackjax_nuts(
        lh,
        pos_init,
        k_nuts,
        n_warmup_steps=100,
        n_samples=5,
    )

    jft.blackjax_nuts(
        lh,
        pos_init,
        k_resume,
        n_samples=5,
        state_and_parameters=state,
    )
