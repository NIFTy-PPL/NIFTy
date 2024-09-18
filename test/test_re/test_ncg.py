#!/usr/bin/env python3

import sys

import pytest

pytest.importorskip("jax")

from functools import partial

import jax.numpy as jnp
from jax import jit, random, value_and_grad
from numpy.testing import assert_allclose

import nifty8.re as jft

pmp = pytest.mark.parametrize


def rosenbrock(np):
    def func(x):
        return jnp.sum(100. * jnp.diff(x)**2 + (1. - x[:-1])**2)

    return func


def himmelblau(np):
    def func(p):
        x, y = p
        return (x**2 + y - 11.)**2 + (x + y**2 - 7.)**2

    return func


def matyas(np):
    def func(p):
        x, y = p
        return 0.26 * (x**2 + y**2) - 0.48 * x * y

    return func


def eggholder(np):
    def func(p):
        x, y = p
        return -(y + 47) * jnp.sin(
            jnp.sqrt(jnp.abs(x / 2. + y + 47.))
        ) - x * jnp.sin(jnp.sqrt(jnp.abs(x - (y + 47.))))

    return func


@pmp("ncg", (jft.newton_cg, jft.static_newton_cg))
def test_ncg_for_pytree(ncg):
    pos = jft.Vector(
        [
            jnp.array(0., dtype=jnp.float32),
            (jnp.array(3., dtype=jnp.float32), ),
            {
                "a": jnp.array(5., dtype=jnp.float32)
            },
        ]
    )
    getters = (lambda x: x[0], lambda x: x[1][0], lambda x: x[2]["a"])
    tgt = [-10., 1., 2.]
    met = [10., 40., 2]

    def model(p):
        losses = []
        for i, get in enumerate(getters):
            losses.append((get(p) - tgt[i])**2 * met[i])
        return jnp.sum(jnp.array(losses))

    def metric(p, tan):
        m = []
        m.append(tan[0] * met[0])
        m.append((tan[1][0] * met[1], ))
        m.append({"a": tan[2]["a"] * met[2]})
        return jft.Vector(m)

    res = ncg(
        fun_and_grad=value_and_grad(model),
        x0=pos,
        hessp=metric,
        maxiter=10,
        absdelta=1e-6
    )
    for i, get in enumerate(getters):
        assert_allclose(get(res), tgt[i], atol=1e-6, rtol=1e-5)


@pmp("seed", (3637, 12, 42))
@pmp("ncg", (jft.newton_cg, jft.static_newton_cg))
def test_ncg(seed, ncg):
    key = random.PRNGKey(seed)
    x = random.normal(key, shape=(3, ))
    diag = jnp.array([1., 2., 3.])
    met = lambda y, t: t / diag
    val_and_grad = lambda y: (
        jnp.sum(y**2 / diag) / 2 - jnp.dot(x, y), y / diag - x
    )

    res = ncg(
        fun_and_grad=val_and_grad,
        x0=x,
        hessp=met,
        maxiter=20,
        absdelta=1e-6,
        name='N'
    )
    assert_allclose(res, diag * x, rtol=1e-4, atol=1e-4)


@pmp("seed", (3637, 12, 42))
def test_static_ncg_jittability(seed):
    key = random.PRNGKey(seed)
    x = random.normal(key, shape=(3, ))
    diag = jnp.array([1., 2., 3.])
    met = lambda y, t: t / diag
    fun = lambda y: jnp.sum(y**2 / diag) / 2 - jnp.dot(x, y)
    grad = lambda y: y / diag - x
    fun_and_grad = lambda y: (fun(y), grad(y))

    for kwargs in [{'fun': fun, 'jac': grad}, {'fun_and_grad': fun_and_grad}]:
        ncg = jit(
            partial(jft.static_newton_cg, hessp=met, **kwargs),
            static_argnames=('name', )
        )
        res = ncg(x0=x, maxiter=20, absdelta=1e-6, name='N')
        assert_allclose(res, diag * x, rtol=1e-4, atol=1e-4)


@pmp("seed", (3637, 12, 42))
@pmp("cg", (jft.cg, jft.static_cg))
def test_cg(seed, cg):
    key = random.PRNGKey(seed)
    sk = random.split(key, 2)
    x = random.normal(sk[0], shape=(3, ))
    # Avoid poorly conditioned matrices by shifting the elements from zero
    diag = 6. + random.normal(sk[1], shape=(3, ))
    mat = lambda x: x / diag

    res, _ = cg(mat, x, resnorm=1e-5, absdelta=1e-5)
    assert_allclose(res, diag * x, rtol=1e-4, atol=1e-4)


@pmp("seed", (3637, 12, 42))
@pmp("cg", (jft.cg, jft.static_cg))
def test_cg_non_pos_def_failure(seed, cg):
    key = random.PRNGKey(seed)
    sk = random.split(key, 2)

    x = random.normal(sk[0], shape=(4, ))
    # Purposely produce a non-positive definite matrix
    diag = jnp.concatenate(
        (jnp.array([-1]), 6. + random.normal(sk[1], shape=(3, )))
    )
    mat = lambda x: x / diag

    with pytest.raises(ValueError):
        _, info = cg(mat, x, resnorm=1e-5, absdelta=1e-5)
        if info < 0:
            raise ValueError()


@pmp("seed", (3637, 12, 42))
def test_cg_steihaug(seed):
    key = random.PRNGKey(seed)
    sk = random.split(key, 2)
    x = random.normal(sk[0], shape=(3, ))
    # Avoid poorly conditioned matrices by shifting the elements from zero
    diag = 6. + random.normal(sk[1], shape=(3, ))
    mat = lambda x: x / diag

    # Note, the solution to the subproblem with infinite trust radius is the CG
    # but with the opposite sign
    res = jft.conjugate_gradient._cg_steihaug_subproblem(
        jnp.nan, -x, mat, resnorm=1e-6, trust_radius=jnp.inf
    )
    assert_allclose(res.step, diag * x, rtol=1e-4, atol=1e-4)


@pmp("seed", (3637, 12, 42))
@pmp("size", (5, 9, 14))
def test_cg_steihaug_vs_cg_consistency(seed, size):
    key = random.PRNGKey(seed)
    sk = random.split(key, 2)

    x = random.normal(sk[0], shape=(size, ))
    # Avoid poorly conditioned matrices by shifting the elements from zero
    mat_val = 6. + random.normal(sk[1], shape=(size, size))
    mat_val = mat_val @ mat_val.T  # Construct a symmetric matrix
    mat = lambda x: mat_val @ x

    # Note, the solution to the subproblem with infinite trust radius is the CG
    # but with the opposite sign
    for i in range(4):
        print(f"Iteration {i:02d}", file=sys.stderr)
        res_cgs = jft.conjugate_gradient._cg_steihaug_subproblem(
            jnp.nan,
            -x,
            mat,
            resnorm=1e-6,
            trust_radius=jnp.inf,
            miniter=i,
            maxiter=i
        )
        res_cg_plain, _ = jft.conjugate_gradient.cg(
            mat, x, resnorm=1e-6, miniter=i, maxiter=i
        )
        assert_allclose(res_cgs.step, res_cg_plain, rtol=1e-4, atol=1e-5)


@pmp("seed", (3637, 12, 42))
def test_static_cg_jittability(seed):
    key = random.PRNGKey(seed)
    sk = random.split(key, 2)
    x = random.normal(sk[0], shape=(3, ))
    # Avoid poorly conditioned matrices by shifting the elements from zero
    diag = 6. + random.normal(sk[1], shape=(3, ))
    mat = lambda x: x / diag

    cg = jit(partial(jft.static_cg, mat=mat))
    res, _ = cg(j=x, x0=None, resnorm=1e-5, absdelta=1e-5)
    assert_allclose(res, diag * x, rtol=1e-4, atol=1e-4)


@pmp(
    "fun_and_init",
    (
        (rosenbrock, jnp.zeros(2)),
        (himmelblau, jnp.zeros(2)),
        (matyas, jnp.ones(2) * 6.),
        # (eggholder, jnp.ones(2) * 100.)  # `eggholder` potential leads to
        # infinite loop in trust-ncg's subproblem solver, see
        # https://github.com/google/jax/issues/15035
    )
)
@pmp("maxiter", (jnp.inf, None))
def test_minimize(fun_and_init, maxiter):
    from jax import grad, hessian
    from scipy.optimize import minimize as opt_minimize

    func, x0 = fun_and_init

    def jft_minimize(x0):
        result = jft.minimize(
            func(jnp),
            x0,
            method='trust-ncg',
            options=dict(
                maxiter=maxiter,
                energy_reduction_factor=None,
                gtol=1e-6,
                initial_trust_radius=1.,
                max_trust_radius=1000.
            ),
        )
        return result.x

    def scp_minimize(x0):
        # Use JAX primitives to take derivates
        fun = func(jnp)
        result = opt_minimize(
            fun, x0, jac=grad(fun), hess=hessian(fun), method='trust-ncg'
        )
        return result.x

    jax_res = jft_minimize(x0)
    scipy_res = scp_minimize(x0)
    assert_allclose(scipy_res, jax_res, rtol=2e-6, atol=2e-5)


if __name__ == "__main__":
    test_ncg_for_pytree()
