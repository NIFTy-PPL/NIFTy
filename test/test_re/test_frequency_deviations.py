#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Author: Matteo Guardiani & Julian Rüstig

import jax.random as random
import numpy as np
import pytest

from nifty8.re.library.mf_model import build_frequency_deviations_model, build_amplitude_model
from nifty8.re.gauss_markov import build_wiener_process

pmp = pytest.mark.parametrize

# wiener_process = build_wiener_process()

@pmp("frequency_deviations_model", [])
@pmp("frequencies", [(0.1,), (0.1, 0.2, 0.3)])
@pmp("reference_frequency_index", [0])
def test_frequency_deviations_init(frequency_deviations_model,
                                   frequencies,
                                   reference_frequency_index):
    pass


@pmp("shape", [(10,), (10, 10)])
@pmp("log_frequencies", [(0.1,), (0.1, 0.2, 0.3)])
@pmp("deviations_settings", [dict(
        process='wiener',
        sigma=(1., 0.1),
    )])
def test_build_frequency_deviations_model(
        shape,
        log_frequencies,
        deviations_settings,
):
    deviations_model = build_frequency_deviations_model(shape,
                                                        log_frequencies,
                                                        0,
                                                        deviations_settings)
    assert deviations_model
    assert deviations_model.domain


# TODO: test that deviations are null at reference frequency
# TODO: test that deviations are slopeless