#!/usr/bin/env python3

import jax.numpy as np
import pytest
from jax import random
from numpy.testing import assert_allclose

import jifty1 as jft

pmp = pytest.mark.parametrize


def lst2fixt(lst):
    @pytest.fixture(params=lst)
    def fixt(request):
        return request.param

    return fixt


def random_noise_std_inv(key, shape):
    diag = 1. / random.exponential(key, shape)

    def noise_std_inv(tangents):
        return diag * tangents

    return noise_std_inv


seed = lst2fixt((3639, 12, 41, 42))
shape = lst2fixt(((4, 2), (2, 1), (5, )))


def test_gaussian_vs_vcgaussian_consistency(seed, shape):
    rtol = 10 * np.finfo(np.zeros(0).dtype).eps
    atol = 1 * np.finfo(np.zeros(0).dtype).eps

    key = random.PRNGKey(seed)
    sk = list(random.split(key, 5))
    d = random.normal(sk.pop(), shape=shape)
    m1 = random.normal(sk.pop(), shape=shape)
    m2 = random.normal(sk.pop(), shape=shape)
    t = random.normal(sk.pop(), shape=shape)
    inv_std = 1. / np.exp(1. + random.normal(sk.pop(), shape=shape))

    gauss = jft.Gaussian(d, noise_std_inv=lambda x: inv_std * x)
    vcgauss = jft.VariableCovarianceGaussian(d)

    diff_g = gauss(m2) - gauss(m1)
    diff_vcg = vcgauss((m2, inv_std)) - vcgauss((m1, inv_std))
    assert_allclose(diff_g, diff_vcg, rtol=rtol, atol=atol)

    met_g = gauss.metric(m1, t)
    met_vcg = vcgauss.metric((m1, inv_std), (t, d / 2))[0]
    assert_allclose(met_g, met_vcg, rtol=rtol, atol=atol)


def test_studt_vs_vcstudt_consistency(seed, shape):
    rtol = 10 * np.finfo(np.zeros(0).dtype).eps
    atol = 4 * np.finfo(np.zeros(0).dtype).eps

    key = random.PRNGKey(seed)
    sk = list(random.split(key, 6))
    d = random.normal(sk.pop(), shape=shape)
    dof = random.normal(sk.pop(), shape=shape)
    m1 = random.normal(sk.pop(), shape=shape)
    m2 = random.normal(sk.pop(), shape=shape)
    t = random.normal(sk.pop(), shape=shape)
    inv_std = 1. / np.exp(1. + random.normal(sk.pop(), shape=shape))

    studt = jft.StudentT(d, dof, noise_std_inv=lambda x: inv_std * x)
    vcstudt = jft.VariableCovarianceStudentT(d, dof)

    diff_t = studt(m2) - studt(m1)
    diff_vct = vcstudt((m2, 1. / inv_std)) - vcstudt((m1, 1. / inv_std))
    assert_allclose(diff_t, diff_vct, rtol=rtol, atol=atol)

    met_g = studt.metric(m1, t)
    met_vcg = vcstudt.metric((m1, 1. / inv_std), (t, d / 2))[0]
    assert_allclose(met_g, met_vcg, rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_gaussian_vs_vcgaussian_consistency(42, (5, ))
