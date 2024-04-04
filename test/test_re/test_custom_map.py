#!/usr/bin/env python3

import pytest

pytest.importorskip("jax")

from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
from numpy.testing import assert_allclose

import nifty8.re as jft

pmp = pytest.mark.parametrize

jax.config.update("jax_enable_x64", True)


def f(u, v):
    return jnp.exp(u @ v)


@pmp("map", (jft.smap, jft.lmap))
@pmp("in_axes", ((0, None), (1, None), (None, 0), (0, 0), (1, 1)))
@pmp("out_axes", (0, ))
@pmp("seed", (32, 42, 43))
def test_map_f(map, in_axes, out_axes, seed):
    fixed_shp = 3
    batched_shp = 4
    key = random.PRNGKey(seed)

    vf = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)
    sf = map(f, in_axes=in_axes, out_axes=out_axes)
    inp = []
    for i, k in zip(in_axes, random.split(key, len(in_axes))):
        if i is None:
            shp = (fixed_shp, )
        elif i == 0:
            shp = (batched_shp, fixed_shp)
        else:
            assert i == 1
            shp = (fixed_shp, batched_shp)
        inp.append(random.normal(k, shape=shp))
    assert_allclose(vf(*inp), sf(*inp), atol=1e-14, rtol=1e-14)


def g(u, v):
    return u, jnp.exp(u @ v)


@pmp("map", (jft.smap, jft.lmap))
@pmp("in_axes,out_axes", (((0, None), (0, 0)), ((None, 1), (None, 0))))
@pmp("seed", (32, 42, 43))
def test_map_g(map, in_axes, out_axes, seed):
    fixed_shp = 3
    batched_shp = 4
    key = random.PRNGKey(seed)

    vf = jax.vmap(g, in_axes=in_axes, out_axes=out_axes)
    sf = map(g, in_axes=in_axes, out_axes=out_axes)
    inp = []
    for i, k in zip(in_axes, random.split(key, len(in_axes))):
        if i is None:
            shp = (fixed_shp, )
        elif i == 0:
            shp = (batched_shp, fixed_shp)
        else:
            assert i == 1
            shp = (fixed_shp, batched_shp)
        inp.append(random.normal(k, shape=shp))
    v = vf(*inp)
    s = sf(*inp)
    tree_map(partial(assert_allclose, atol=1e-14, rtol=1e-14), v, s)


def h(u, v, w):
    return v + w.sum() - u.sum(), w * jnp.sin(u.sum() * v.sum()
                                             ), u + v.sum() * w.sum()


@pmp("map", (jft.smap, jft.lmap))
@pmp("in_axes", (0, 1, 2))
@pmp("out_axes", (0, 1, 2))
@pmp("seed", (32, 42, 43))
def test_map_h(map, in_axes, out_axes, seed):
    key = random.PRNGKey(seed)

    vf = jax.vmap(h, in_axes=in_axes, out_axes=out_axes)
    sf = map(h, in_axes=in_axes, out_axes=out_axes)
    inp = [random.normal(k, shape=(1, 2, 3)) for k in random.split(key, 3)]
    v = vf(*inp)
    s = sf(*inp)
    tree_map(partial(assert_allclose, atol=1e-14, rtol=1e-14), v, s)
