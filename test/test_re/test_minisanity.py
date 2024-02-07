#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

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


@pmp("seed", (42, 43))
@pmp("complex", (True, False))
def test_reduced_chi_square_dtype(seed, complex):
    sseq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(sseq)
    n_npix = 500
    norm_res = rng.normal(size=(n_npix), )
    if complex:
        norm_res = norm_res + 1j * rng.normal(size=(n_npix), )
    rrs = reduced_residual_stats(norm_res)
    assert rrs.mean.dtype == norm_res.dtype
    # chi^2 should be real and positive
    assert np.abs(rrs.reduced_chisq[0]) == rrs.reduced_chisq[0]

    atol, rtol = 1e-1, 1e-1
    assert_allclose(rrs.reduced_chisq[0], 1., atol=atol, rtol=rtol)
    assert_allclose(rrs.mean, 0., atol=atol, rtol=rtol)


if __name__ == "__main__":
    test_reduced_residual_stats_normal_residuals(33)
    test_reduced_chi_square_dtype(42, True)
