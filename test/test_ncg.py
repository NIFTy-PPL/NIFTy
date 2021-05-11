#!/usr/bin/env python3

import jax.numpy as np
import jifty1 as jft
from jax import value_and_grad
from numpy import testing


def test_ncg():
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
        pos, value_and_grad(model), metric, iterations=10, absdelta=1e-6
    )
    for i, get in enumerate(getters):
        testing.assert_allclose(get(res), tgt[i], atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    test_ncg()
