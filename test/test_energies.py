#!/usr/bin/env python3

import jax.numpy as np
import pytest
from functools import partial
from jax import eval_shape, random
from jax.tree_util import tree_map, tree_leaves
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
lh_init_true = (
    (
        jft.Gaussian, {
            "data": random.normal,
            "noise_std_inv": random_noise_std_inv
        }, None
    ), (
        jft.StudentT, {
            "data": random.normal,
            "dof": random.exponential,
            "noise_std_inv": random_noise_std_inv
        }, None
    ), (
        jft.Poissonian, {
            "data": partial(random.poisson, lam=3.14)
        }, random.exponential
    )
)
lh_init_approx = (
    (
        jft.VariableCovarianceGaussian, {
            "data": random.normal
        }, lambda key, shape: (
            random.normal(key, shape=shape), 1. / np.
            exp(random.normal(key, shape=shape))
        )
    ), (
        jft.VariableCovarianceStudentT, {
            "data": random.normal,
            "dof": random.exponential
        }, lambda key, shape: (
            random.normal(key, shape=shape),
            np.exp(1. + random.normal(key, shape=shape))
        )
    )
)


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


@pmp("lh_init", lh_init_true + lh_init_approx)
def test_left_sqrt_metric_vs_metric_consistency(seed, shape, lh_init):
    rtol = 4 * np.finfo(np.zeros(0).dtype).eps
    atol = 0.
    aallclose = partial(assert_allclose, rtol=rtol, atol=atol)

    N_TRIES = 5

    lh_init_method, draw, latent_init = lh_init
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, 1 + len(draw))
    init_kwargs = {
        k: method(key=sk, shape=shape)
        for (k, method), sk in zip(draw.items(), subkeys)
    }
    lh = lh_init_method(**init_kwargs)

    energy, lsm, lsm_shp = lh.energy, lh.left_sqrt_metric, lh.lsm_tangents_shape
    # Let JIFTy infer the metric from the left-square-root-metric
    lh_mini = jft.Likelihood(
        energy, left_sqrt_metric=lsm, lsm_tangents_shape=lsm_shp
    )

    rng_method = latent_init if latent_init is not None else random.normal
    for _ in range(N_TRIES):
        key, *sk = random.split(key, 3)
        p = rng_method(sk.pop(), shape=shape)
        t = rng_method(sk.pop(), shape=shape)
        tree_map(aallclose, lh.metric(p, t), lh_mini.metric(p, t))


@pmp("lh_init", lh_init_true)
def test_transformation_vs_left_sqrt_metric_consistency(seed, shape, lh_init):
    rtol = 4 * np.finfo(np.zeros(0).dtype).eps
    atol = 0.

    N_TRIES = 5

    lh_init_method, draw, latent_init = lh_init
    key = random.PRNGKey(seed)
    key, *subkeys = random.split(key, 1 + len(draw))
    init_kwargs = {
        k: method(key=sk, shape=shape)
        for (k, method), sk in zip(draw.items(), subkeys)
    }
    lh = lh_init_method(**init_kwargs)
    if lh._transformation is None:
        pytest.skip("no transformation rule implemented yet")

    energy, lsm, lsm_shp = lh.energy, lh.left_sqrt_metric, lh.lsm_tangents_shape
    # Let JIFTy infer the left-square-root-metric and the metric from the
    # transformation
    lh_mini = jft.Likelihood(
        energy, left_sqrt_metric=lsm, lsm_tangents_shape=lsm_shp
    )

    rng_method = latent_init if latent_init is not None else random.normal
    for _ in range(N_TRIES):
        key, *sk = random.split(key, 3)
        p = rng_method(sk.pop(), shape=shape)
        t = rng_method(sk.pop(), shape=shape)
        assert_allclose(
            lh.left_sqrt_metric(p, t),
            lh_mini.left_sqrt_metric(p, t),
            rtol=rtol,
            atol=atol
        )
        assert_allclose(
            lh.metric(p, t), lh_mini.metric(p, t), rtol=rtol, atol=atol
        )


if __name__ == "__main__":
    test_gaussian_vs_vcgaussian_consistency(42, (5, ))
