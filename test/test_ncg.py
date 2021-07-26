#!/usr/bin/env python3

import jax.numpy as np
import jifty1 as jft
import pytest
from jax import value_and_grad, random
from numpy.testing import assert_allclose

pmp = pytest.mark.parametrize


def test_ncg_for_pytree():
    pos = jft.Field([np.array(0.), (np.array(3.), ), {"a": np.array(5.)}])
    getters = (lambda x: x[0], lambda x: x[1][0], lambda x: x[2]["a"])
    tgt = [-10., 1., 2.]
    met = [10., 40., 2]

    def model(p):
        losses = []
        for i, get in enumerate(getters):
            losses.append((get(p) - tgt[i])**2 * met[i])
        return np.sum(np.array(losses))

    def metric(p, tan):
        m = []
        m.append(tan[0] * met[0])
        m.append((tan[1][0] * met[1], ))
        m.append({"a": tan[2]["a"] * met[2]})
        return jft.Field(m)

    res = jft.newton_cg(
        value_and_grad(model), pos, metric, maxiter=10, absdelta=1e-6
    )
    for i, get in enumerate(getters):
        assert_allclose(get(res), tgt[i], atol=1e-6, rtol=1e-5)


@pmp("seed", (3637, 12, 42))
def test_ncg(seed):
    key = random.PRNGKey(seed)
    x = random.normal(key, shape=(3, ))
    diag = np.array([1., 2., 3.])
    met = lambda y, t: t / diag
    val_and_grad = lambda y: (
        np.sum(y**2 / diag) / 2 - np.dot(x, y), y / diag - x
    )

    res = jft.newton_cg(val_and_grad, x, met, 20, absdelta=1e-6, name='N')
    assert_allclose(res, diag * x, rtol=1e-4, atol=1e-4)


@pmp("seed", (3637, 12, 42))
def test_cg(seed):
    key = random.PRNGKey(seed)
    sk = random.split(key, 2)
    x = random.normal(sk[0], shape=(3, ))
    # Avoid poorly conditioned matrices by shifting the elements from zero
    diag = 6. + random.normal(sk[1], shape=(3, ))
    mat = lambda x: x / diag

    res, _ = jft.cg(mat, x, resnorm=1e-5, absdelta=1e-5)
    assert_allclose(res, diag * x, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_ncg_for_pytree()
