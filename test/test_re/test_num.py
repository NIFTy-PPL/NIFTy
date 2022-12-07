#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import numpy as np
from numpy.testing import assert_allclose
import pytest

from nifty8.re.num import unique

pmp = pytest.mark.parametrize


def test_unique_known_in_out():
    t0 = np.arange(4).reshape(2, 2)
    t1 = 1 + t0
    t1_1 = (1. + 1e-8) + t0
    t2 = 2. + t0
    test = np.stack([t0, t0, t1, t1, t1_1, t2, t1_1, t1, t0], axis=-1)

    u = unique(test, axis=-1)
    assert u.shape[-1] == 3

    atol, rtol = 1e-12, 1e-12
    u, i = unique(test, atol=atol, rtol=rtol, axis=-1, return_inverse=True)
    assert u.shape[-1] == 4
    assert_allclose(
        test, np.take(u, i, axis=-1), atol=1.01 * atol, rtol=1.01 * rtol
    )


@pmp("seed", (42, 43))
@pmp("shape", ((5, ), (8, 8, 13), (2, 13)))
def test_unique_fuzz(seed, shape):
    sseq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(sseq)
    atol, rtol = 1e-5, 1e-4

    d = rng.normal(size=shape)
    for axis in range(d.ndim):
        da = np.repeat(d, rng.integers(1, 5, size=d.shape[axis]), axis=axis)
        da = rng.permutation(da, axis=axis)
        u, i = unique(da, atol=atol, rtol=rtol, axis=-1, return_inverse=True)
        assert np.allclose(
            da, np.take(u, i, axis=-1), atol=1.01 * atol, rtol=1.01 * rtol
        )


if __name__ == "__main__":
    test_unique_known_in_out()
    test_unique_fuzz(42, (8, 8, 13))
