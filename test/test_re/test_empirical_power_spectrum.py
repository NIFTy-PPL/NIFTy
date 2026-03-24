#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import jax
import jax.numpy as jnp
import nifty.re as jft
import pytest
from numpy.testing import assert_allclose

jax.config.update("jax_enable_x64", True)

pmp = pytest.mark.parametrize


@pmp("loglogavgslope", [-1, -2, -3, -4, -5, -6])
def test_empirical_ps_computation_with_power_law_spectrum_cf(loglogavgslope):
    n_samples = 10
    key = jax.random.PRNGKey(42)
    distances = 1.0

    cfm = jft.CorrelatedFieldMaker(prefix="")
    cfm.set_amplitude_total_offset(offset_mean=0.0, offset_std=(1.0, 1e-16))
    cfm.add_fluctuations(
        shape=(int(2**16)),
        distances=distances,
        fluctuations=(1.0, 1e-16),
        loglogavgslope=(loglogavgslope, 1e-16),
        asperity=None,
        flexibility=None,
        non_parametric_kind="power",
    )

    cf = cfm.finalize()
    amp = cfm.fluctuations[0]
    cfv = jft.VModel(cf, axis_size=n_samples, in_axes="xi")

    xi = cfv.init(key)
    cf_samples = cfv(xi)

    ps_cf = jnp.float_power(amp(xi), 2)
    k_bin_centers_cf = 2.0 * jnp.pi * amp.grid.harmonic_grid.mode_lengths

    ps_emp, k_bin_centers_emp = jft.compute_empirical_power_spectrum(
        field=cf_samples,
        axes=1,
        distances=distances,
        n_bins=16,  # use few bins to get a smooth power spectrum estimate
    )

    median_ps_emp = jnp.median(ps_emp, axis=0)

    # fit linear functions to power spectra, compare coefficients
    def regress_ps(k, ps):
        log_k = jnp.log10(k)
        log_P = jnp.log10(ps)
        coeff = jnp.polyfit(log_k, log_P, deg=1)
        return coeff

    coeff_gt = regress_ps(k_bin_centers_cf[1:], ps_cf[1:])
    coeff_emp = regress_ps(k_bin_centers_emp, median_ps_emp)

    assert_allclose(coeff_emp, coeff_gt, rtol=0.1)
