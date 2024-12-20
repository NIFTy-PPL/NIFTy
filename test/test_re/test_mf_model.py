#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani & Julian Rüstig

import jax.random as random
import pytest
from numpy.testing import assert_allclose
import numpy as np

import nifty8.re as jft
from nifty8.re.correlated_field import (
    make_grid, NonParametricAmplitude, MaternAmplitude)
from nifty8.re.library.frequency_deviations import build_frequency_deviations_model_with_degeneracies
from nifty8.re.library.spectral_behaviour import SpectralIndex
from nifty8.re.library.mf_spectral_fun import _build_fluctuations_model

pmp = pytest.mark.parametrize

# TODO: reintroduce this test when amplitudes are implemented
# @pmp("shape", [(10,), (10, 10)])
# @pmp("settings, amplitude_model",
#     [
#         (dict(
#             fluctuations=(1.0, 0.02),
#             loglogavgslope=(-2.0, 0.1),
#             flexibility=None,
#             asperity=None,),
#          "non_parametric",
#         ),
#         (dict(
#             scale=(1.0, 0.02),
#             cutoff=(0.1, 0.01),
#             loglogslope=(-4, 0.1),),
#         "matern")
#     ]
# )
# @pmp("renormalize_amplitude", [True, False])
# @pmp("prefix", ["test", "", None])
# @pmp("kind", ["amplitude", "power"])
# def test_build_amplitude_model(
#     shape,
#     settings,
#     amplitude_model,
#     renormalize_amplitude,
#     prefix,
#     kind,
# ):
#     dist = .1
#     grid = _make_grid(shape, dist, "fourier")
#     amplitude = build_amplitude_model(grid,
#                                       settings,
#                                       amplitude_model,
#                                       renormalize_amplitude,
#                                       prefix,
#                                       kind,
#                                       )
#     assert amplitude
#     assert amplitude.domain


@pmp("shape", [(10,), (10, 10)])
@pmp("distances", [0.1])
@pmp("log_frequencies", [np.array((0.1,))])
@pmp("reference_frequency_index", [0])
@pmp("zero_mode", [jft.Model(lambda p: 0., domain={"zero_mode": None})])
@pmp("spectral_index_mean", [jft.NormalPrior(0., 1., name="spectral_index_mean")])
@pmp("spectral_index_fluctuations", [jft.LogNormalPrior(0.1, 1., name="spectral_index_fluctuations")])
@pmp("deviations_settings", [dict(process='wiener', sigma=(1., 0.1),), None])
def test_correlated_multi_frequency_sky_init(
    shape,
    distances,
    log_frequencies,
    reference_frequency_index,
    zero_mode,
    spectral_index_mean,
    spectral_index_fluctuations,
    deviations_settings,
):
    grid = make_grid(shape, distances, "fourier")
    spatial_amplitude = NonParametricAmplitude(
        grid, jft.lognormal_prior(.1, .01), None)

    spatial_fluctuations = jft.LogNormalPrior(
        0.1, 10., name="spatial_fluctuations")

    spectral_behavior = SpectralIndex(
        log_frequencies=log_frequencies,
        mean=spectral_index_mean,
        fluctuations=spectral_index_fluctuations,
        reference_frequency_index=reference_frequency_index,
    )

    deviations_model = build_frequency_deviations_model_with_degeneracies(
        shape, log_frequencies, reference_frequency_index, deviations_settings)

    mf_sky = jft.CorrelatedMultiFrequencySky(
        zero_mode,
        spatial_fluctuations,
        spatial_amplitude,
        spectral_behavior,
        spectral_index_deviations=deviations_model,
    )

    assert mf_sky
    assert mf_sky.domain


@pmp("seed", [12, 42])
@pmp("shape", [(10,), (10, 10)])
@pmp("distances", [0.1])
@pmp("zero_mode", [jft.Model(lambda p: 0.1, domain={"test_zero_mode": None})])
@pmp("zero_mode_offset", [0.])
@pmp("spectral_index_mean", [jft.NormalPrior(0., 1., name="spectral_index_mean")])
@pmp("spectral_index_fluctuations", [jft.LogNormalPrior(0.1, 1., name="spectral_index_fluctuations")])
def test_spatial_convolution(
    seed,
    shape,
    distances,
    zero_mode,
    zero_mode_offset,
    spectral_index_mean,
    spectral_index_fluctuations,
):
    grid = make_grid(shape, distances, "fourier")
    prefix = "test"
    fluct = (1, 0.01)
    avgsl = (-4., .1)

    spatial_amplitude = NonParametricAmplitude(
        grid, jft.normal_prior(*avgsl), None, prefix=f"{prefix}_")

    spatial_fluctuations = _build_fluctuations_model(
        prefix=f'{prefix}_spatial',
        fluctuation_settings=fluct,
        shape=shape,
    )

    spectral_behavior = SpectralIndex(
        log_frequencies=np.array((0.,)),
        mean=spectral_index_mean,
        fluctuations=spectral_index_fluctuations,
        reference_frequency_index=0,
    )

    mf_sky = jft.CorrelatedMultiFrequencySky(
        zero_mode,
        spatial_fluctuations,
        spatial_amplitude,
        spectral_behavior,
        nonlinearity=lambda x: x,
    )

    cfm = jft.CorrelatedFieldMaker("")
    cfm.set_amplitude_total_offset(zero_mode_offset, zero_mode)
    cfm.add_fluctuations(shape,
                         distances,
                         fluct,
                         avgsl)
    cfm = cfm.finalize()
    rp = mf_sky.init(random.PRNGKey(seed))
    cfm_rp = {"zeromode": rp[f"{prefix}_zero_mode"],
              "fluctuations": rp[f"{prefix}_spatial_fluctuations"],
              "loglogavgslope": rp[f"{prefix}_loglogavgslope"],
              "xi": rp[f"{prefix}_spatial_xi"],
              }
    zm_index = tuple(0 for _ in grid.shape)
    # this happens because the cfm has a degeneracy in the
    # zero mode parametrization
    diff = zero_mode(rp) * (1-rp[f"{prefix}_spatial_xi"][zm_index])
    assert_allclose(mf_sky(rp)[0]-diff, cfm(cfm_rp))


@pmp("seed", [12, 42])
@pmp("shape", [(10,), (10, 10)])
@pmp("distances", [0.1])
@pmp("log_frequencies", [(0.1, 0.2, 0.6)])
@pmp("zero_mode", [jft.Model(lambda p: 0.1, domain={"zero_mode": None})])
@pmp("zero_mode_offset", [0.])
@pmp("spectral_index_mean", [jft.NormalPrior(0., 1.,
                                             name="spectral_index_mean")]
     )
@pmp("spectral_index_fluctuations", [(0.1, 1.)])
def test_apply_with_and_without_frequency_deviations(
    seed,
    shape,
    distances,
    log_frequencies,
    zero_mode,
    zero_mode_offset,
    spectral_index_mean,
    spectral_index_fluctuations,
):
    grid = make_grid(shape, distances, "fourier")
    reference_freq_idx = 0
    prefix = "test"
    fluct = (0.1, 0.01)
    avgsl = (-4., .1)
    spatial_amplitude = NonParametricAmplitude(grid,
                                               jft.normal_prior(*avgsl),
                                               None,
                                               prefix=f"{prefix}_",
                                               )

    log_frequencies = np.array(log_frequencies)

    spatial_fluctuations = _build_fluctuations_model(
        prefix=f'{prefix}_spatial',
        fluctuation_settings=fluct,
        shape=shape,
    )
    spectral_fluctuations = _build_fluctuations_model(
        prefix=f'{prefix}_spatial',
        fluctuation_settings=spectral_index_fluctuations,
        shape=shape,
    )

    spectral_behavior = SpectralIndex(
        log_frequencies=log_frequencies,
        mean=spectral_index_mean,
        fluctuations=spectral_fluctuations,
        reference_frequency_index=reference_freq_idx,
    )

    mf_sky_wo_deviations = jft.CorrelatedMultiFrequencySky(
        zero_mode,
        spatial_fluctuations,
        spatial_amplitude,
        spectral_behavior,
    )

    deviations_settings = dict(
        process='wiener',
        sigma=(1.e-15, 1.e-18),
    )

    deviations_model = build_frequency_deviations_model_with_degeneracies(
        shape, log_frequencies, 0, deviations_settings)

    mf_sky = jft.CorrelatedMultiFrequencySky(
        zero_mode,
        spatial_fluctuations,
        spatial_amplitude,
        spectral_behavior,
        spectral_index_deviations=deviations_model
    )

    rp = mf_sky.init(random.PRNGKey(seed))
    assert_allclose(mf_sky_wo_deviations(rp), mf_sky(rp))
