#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial

import jax
import pytest
from jax import random
from numpy.testing import assert_allclose

import nifty8.re as jft
from nifty8.re import partial_insert_and_remove as jpartial

pmp = pytest.mark.parametrize


def _identity(x):
    return x


def test_partial_insert_and_remove():
    # _identity takes exactly one argument, thus `insert_axes` and `flat_fill`
    # are length one tuples
    _id_part = jpartial(
        _identity,
        insert_axes=(jft.Vector({
            "a": (True, False),
            "b": False
        }), ),
        flat_fill=(("THIS IS input['a'][0]", ), )
    )
    out = _id_part(("THIS IS input['a'][1]", "THIS IS input['b']"))
    assert out == jft.Vector(
        {
            "a": ("THIS IS input['a'][0]", "THIS IS input['a'][1]"),
            "b": "THIS IS input['b']"
        }
    )


@pmp("seed", (33, 42, 43))
def test_likelihood_partial(seed):
    atol, rtol = 1e-14, 1e-14
    aallclose = partial(assert_allclose, rtol=rtol, atol=atol)

    key = random.PRNGKey(seed)

    domain = jft.Vector(
        {
            "a": jft.ShapeWithDtype(128),
            "b": jft.ShapeWithDtype(64)
        }
    )
    key, sk_d, sk_p = random.split(key, 3)
    data = jft.random_like(sk_d, domain)
    primals = jft.random_like(sk_p, domain)

    gaussian = jft.Gaussian(data)
    aallclose(gaussian(data), 0., atol=atol)
    gaussian_part, primals_liquid = gaussian.freeze(("b", ), primals)
    assert primals_liquid.tree[0].shape == domain["a"].shape
    aallclose(gaussian_part(primals_liquid), gaussian(primals))
    jax.tree_map(
        aallclose,
        gaussian_part.left_sqrt_metric(primals_liquid, data).tree[0],
        gaussian.left_sqrt_metric(primals, data).tree["a"],
    )
    jax.tree_map(
        aallclose,
        gaussian_part.right_sqrt_metric(primals_liquid, primals_liquid),
        gaussian.right_sqrt_metric(primals, primals),
    )
    aallclose(
        gaussian_part.metric(primals_liquid, primals_liquid).tree[0],
        gaussian.metric(primals, primals).tree["a"],
    )


if __name__ == "__main__":
    test_likelihood_partial(33)
