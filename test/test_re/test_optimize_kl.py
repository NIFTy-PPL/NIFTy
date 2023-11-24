#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from jax import random
from nifty8.re.optimize_kl import concatenate_zip

pmp = pytest.mark.parametrize


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
