#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import jax
import numpy as np
import pytest
from jax import random
from jax.scipy.ndimage import map_coordinates

from nifty8.re import SamplingCartesianGridLOS

jax.config.update("jax_enable_x64", True)

pmp = pytest.mark.parametrize


@pmp("seed", (42,))
@pmp("shape", ((10,), (25,), (12, 12), (6, 6, 6)))
def test_sampling_cartesian_grid_los(seed, shape):
    key = random.PRNGKey(seed)
    n_test_points = 1_000
    test_los = SamplingCartesianGridLOS(
        np.zeros((len(shape),)),
        np.ones((len(shape),))[np.newaxis],
        distances=tuple(1.0 / s for s in shape),
        shape=shape,
        n_sampling_points=n_test_points,
    )
    key, sk = random.split(key)
    for test_x in (np.ones(shape), random.normal(sk, shape)):
        desired = map_coordinates(
            test_x,
            [
                np.linspace(0, s - 1, num=n_test_points, endpoint=False)
                for s in test_x.shape
            ],
            order=1,
            cval=np.nan,
        ).mean()
        desired *= np.linalg.norm(np.ones((len(shape),)))
        np.testing.assert_allclose(
            test_los(test_x).sum(),
            desired,
            rtol=1e-2,
        )
