#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import jax
import jax.numpy as jnp
import jax.random as random
import pytest
from jax.tree_util import tree_map
from numpy.testing import assert_allclose

import nifty8.re as jft

jax.config.update("jax_enable_x64", True)

pmp = pytest.mark.parametrize


def _v_compare(model, axis_size, in_axes=0, out_axes=0):
    vmodel = jft.VModel(model, axis_size=axis_size, in_axes=in_axes, out_axes=out_axes)

    x = jft.random_like(random.PRNGKey(10), vmodel.domain)
    res = vmodel(x)
    gt = jax.vmap(model, in_axes=(in_axes,), out_axes=out_axes)(x)

    tree_map(assert_allclose, gt, res)


def test_base():
    dummy = jft.WrappedCall(jnp.exp, name="xi", shape=(3,), white_init=True)

    _v_compare(dummy, axis_size=5)


def test_multi():
    tree = (
        {"a": jnp.zeros(3, dtype=float), "b": jnp.zeros(5, dtype=float)},
        jnp.zeros(5, dtype=float),
    )
    in_axes = ({"a": None, "b": 0}, None)
    out_axes = (0, 1)

    class Mymodel(jft.Model):
        def __init__(self, tree):
            domain = tree
            super().__init__(domain=domain, white_init=True)

        def __call__(self, x):
            res1 = x[0]["a"].sum()
            res2 = x[1] + x[0]["b"]
            return (res1, res2)

    mymodel = Mymodel(tree)

    _v_compare(mymodel, axis_size=4, in_axes=in_axes, out_axes=out_axes)


@pmp("key", ["a", ("a",), ["a"]])
def test_key(key):
    def f(x):
        return x["a"] + x["b"]

    model = jft.Model(
        f,
        domain={
            "a": jft.ShapeWithDtype((3,), float),
            "b": jft.ShapeWithDtype((1,), float),
        },
        white_init=True,
    )

    vmodel = jft.VModel(model, axis_size=10, in_axes=key)

    x = jft.random_like(random.PRNGKey(10), vmodel.domain)
    res = vmodel(x)
    gt = jax.vmap(model, in_axes=({"a": 0, "b": None},))(x)
    assert_allclose(gt, res)

    # Check intended dict handling with dummy input
    x = x | {"dummy": 5.0}
    res = vmodel(x)
    assert_allclose(gt, res)
