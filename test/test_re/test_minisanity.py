#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from nifty8.re.minisanity import reduced_residual_stats

pmp = pytest.mark.parametrize


@pmp("seed", (33, 42, 43))
def test_reduced_residual_stats_normal_residuals(seed):
    sseq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(sseq)
    ndof = int(1e+3)
    atol, rtol = 1e-14, 1e-14

    normalized_residuals = rng.normal(size=(ndof, ))

    rrs = reduced_residual_stats(normalized_residuals)
    assert_array_equal(rrs.ndof, ndof)
    assert_allclose(
        rrs.mean[0], np.mean(normalized_residuals), atol=atol, rtol=rtol
    )
    assert_allclose(
        rrs.reduced_chisq[0],
        np.sum(normalized_residuals**2) / rrs.ndof,
        atol=atol,
        rtol=rtol
    )


if __name__ == "__main__":
    test_reduced_residual_stats_normal_residuals(33)
