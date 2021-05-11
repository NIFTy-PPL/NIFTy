#!/usr/bin/env python3

import jax.numpy as np
import jifty1 as jft
import pytest
from jax import random
from numpy.testing import assert_allclose

pmp = pytest.mark.parametrize


@pmp("seed", (3637, 12, 42))
@pmp("shape", ((4, 2), (2, 1), (5, )))
def test_gaussian_vs_vcgaussian_consistency(seed, shape):
    key = random.PRNGKey(seed)
    sk = random.split(key, 5)
    d = random.normal(sk[0], shape=shape)
    m1 = random.normal(sk[1], shape=shape)
    m2 = random.normal(sk[2], shape=shape)
    t = random.normal(sk[3], shape=shape)
    inv_std = 1. / np.exp(1. + random.normal(sk[4], shape=shape))

    gauss = jft.Gaussian(d, noise_std_inv=lambda x: x * inv_std)
    vcgauss = jft.VariableCovarianceGaussian(d)

    diff_g = gauss(m2) - gauss(m1)
    diff_vcg = vcgauss((m2, inv_std)) - vcgauss((m1, inv_std))
    assert_allclose(diff_g, diff_vcg, rtol=1e-05, atol=1e-08)

    met_g = gauss.metric(m1, t)
    met_vcg = vcgauss.metric((m1, inv_std), (t, d / 2))[0]
    assert_allclose(met_g, met_vcg, rtol=1e-05, atol=1e-08)


if __name__ == "__main__":
    test_gaussian_vs_vcgaussian_consistency(42, (5, ))
