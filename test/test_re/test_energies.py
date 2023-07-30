#!/usr/bin/env python3

import pytest
pytest.importorskip("jax")

import jax.numpy as jnp
from functools import partial
from jax import random
from jax.tree_util import tree_map
from numpy.testing import assert_allclose

import nifty8.re as jft
import nifty8 as ift

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
            random.normal(key, shape=shape), 1. / jnp.
            exp(random.normal(key, shape=shape))
        )
    ), (
        jft.VariableCovarianceStudentT, {
            "data": random.normal,
            "dof": random.exponential
        }, lambda key, shape: (
            random.normal(key, shape=shape),
            jnp.exp(1. + random.normal(key, shape=shape))
        )
    )
)


def test_gaussian_vs_vcgaussian_consistency(seed, shape):
    rtol = 10 * jnp.finfo(jnp.zeros(0).dtype).eps
    atol = 5 * jnp.finfo(jnp.zeros(0).dtype).eps

    key = random.PRNGKey(seed)
    sk = list(random.split(key, 5))
    d = random.normal(sk.pop(), shape=shape)
    m1 = random.normal(sk.pop(), shape=shape)
    m2 = random.normal(sk.pop(), shape=shape)
    t = random.normal(sk.pop(), shape=shape)
    inv_std = 1. / jnp.exp(1. + random.normal(sk.pop(), shape=shape))

    gauss = jft.Gaussian(d, noise_std_inv=lambda x: inv_std * x)
    vcgauss = jft.VariableCovarianceGaussian(d)

    diff_g = gauss(m2) - gauss(m1)
    diff_vcg = vcgauss((m2, inv_std)) - vcgauss((m1, inv_std))
    assert_allclose(diff_g, diff_vcg, rtol=rtol, atol=atol)

    met_g = gauss.metric(m1, t)
    met_vcg = vcgauss.metric((m1, inv_std), (t, d / 2))[0]
    assert_allclose(met_g, met_vcg, rtol=rtol, atol=atol)


def test_studt_vs_vcstudt_consistency(seed, shape):
    rtol = 10 * jnp.finfo(jnp.zeros(0).dtype).eps
    atol = 4 * jnp.finfo(jnp.zeros(0).dtype).eps

    key = random.PRNGKey(seed)
    sk = list(random.split(key, 6))
    d = random.normal(sk.pop(), shape=shape)
    dof = random.normal(sk.pop(), shape=shape)
    m1 = random.normal(sk.pop(), shape=shape)
    m2 = random.normal(sk.pop(), shape=shape)
    t = random.normal(sk.pop(), shape=shape)
    inv_std = 1. / jnp.exp(1. + random.normal(sk.pop(), shape=shape))

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
    rtol = 4 * jnp.finfo(jnp.zeros(0).dtype).eps
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
    rtol = 4 * jnp.finfo(jnp.zeros(0).dtype).eps
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

@pmp('cpx', [False, True])
def test(cpx):
    sp = ift.RGSpace(1)
    if cpx :
        val = 1.+1.j
        data = ift.full(sp, 0+0j)
    else:
        val = 1.
        data = ift.full(sp, 0)


    fl = ift.full(sp, val)
    rls = ift.Realizer(sp)
    res = ift.Adder(data, neg=True) @ ift.makeOp(fl) @ rls.adjoint
    res = res.ducktape('res')
    res = res.ducktape_left('res')
    invcov = ift.exp(ift.makeOp(ift.full(sp, 1.)))
    invcov = invcov.ducktape('invcov')
    invcov = invcov.ducktape_left('invcov')
    op = res + invcov
    if cpx:
        dt = jnp.complex128
    else:
        dt = jnp.float64
    varcov = ift.VariableCovarianceGaussianEnergy(sp, 'res', 'invcov', dt) @ op


    def op_jft(x):
        return [val*x['res'], jnp.sqrt(jnp.exp(x['invcov']))]
    varcov_jft = jft.VariableCovarianceGaussian(data.val, cpx) @ op_jft

    # test val
    inp = ift.from_random(op.domain)
    lh0 = varcov(inp)
    lh1 = varcov_jft(inp.val)
    print("test val: ")
    print(f"nifty: {lh0.val}")
    print(f"jft: {lh1}")
    print("")
    assert_allclose(lh0.val, lh1)

    # test metric
    lin = varcov(ift.Linearization.make_var(inp, want_metric=True))
    met = lin.metric
    inp2 = ift.from_random(met.domain)
    met_res = met(inp2)
    met_res_jax = varcov_jft.metric(inp.val, inp2.val)
    print("test metric: ")
    print(f"nifty: {met_res.val}")
    print(f"jft: {met_res_jax}")
    print("")
    assert_allclose(met_res['invcov'].val, met_res_jax['invcov'])
    assert_allclose(met_res['res'].val, met_res_jax['res'])

    # test transform
    inp3 = ift.from_random(varcov.get_transformation()[1].domain)
    res1 = varcov.get_transformation()[1](inp3)
    res2 = varcov_jft.transformation(inp3.val)
    print("test transform: ")
    print(f"nifty: {res1.val}")
    print(f"jft: {res2}")
    print("")
    assert_allclose(res1['res'].val, res2[0])
    assert_allclose(res1['invcov'].val, res2[1])

if __name__ == "__main__":
    test_gaussian_vs_vcgaussian_consistency(42, (5, ))
