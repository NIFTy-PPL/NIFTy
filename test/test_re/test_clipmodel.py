#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Author: Jakob Roth

import jax
import jax.numpy as jnp
import jax.random as random
import pytest

from functools import partial

import nifty.re as jft

jax.config.update("jax_enable_x64", True)

pmp = pytest.mark.parametrize


@pmp(
    "pytree_shape",
    (
        jax.ShapeDtypeStruct((3, 5), float),
        {
            "a": jax.ShapeDtypeStruct((2, 4), float),
            "b": [
                jax.ShapeDtypeStruct((1, 3), float),
                jax.ShapeDtypeStruct((2,), float),
            ],
        },
    ),
)
@pmp("warn", (True, False))
@pmp("use_custom_clip", (True, False))
def test_key(pytree_shape, warn, use_custom_clip):
    threshold = 7.2
    seed = 42
    key = random.PRNGKey(seed)
    inp = 20 * jft.Vector(jft.random_like(key, pytree_shape))
    if use_custom_clip:
        custom_clip_func = partial(jnp.clip, min=-threshold, max=threshold)
    else:
        custom_clip_func = None

    model = jft.Model(lambda x: x, domain=pytree_shape)
    clip_model = jft.ClipModel(
        model=model, threshold=threshold, warn=warn, custom_clip_func=custom_clip_func
    )
    inp_clip = clip_model(inp)
    larger_than_t = (inp_clip > threshold).sum()
    smaller_than_t = (inp_clip < -threshold).sum()

    assert larger_than_t == 0
    assert smaller_than_t == 0
