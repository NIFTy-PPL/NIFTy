#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from typing import Iterable
import pytest


import jax.random as random
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose
from jax import vmap

import nifty8.re as jft


def _v_compare(model, axis_size, in_axes=0, out_axes=0):
    vmodel = jft.VModel(model, axis_size=axis_size, in_axes=in_axes, out_axes=out_axes)

    x = jft.random_like(random.PRNGKey(10), vmodel.domain)
    res = vmodel(x)
    gt = vmap(model, in_axes=(in_axes,), out_axes=out_axes)(x)

    if isinstance(gt, tuple):
        for gg, rr in zip(gt, res):
            assert_allclose(gg, rr)
    else:
        assert_allclose(gt, res)


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
