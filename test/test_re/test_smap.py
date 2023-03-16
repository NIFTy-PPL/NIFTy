#!/usr/bin/env python3

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
from numpy.testing import assert_array_equal

import nifty8.re as jft

pmp = pytest.mark.parametrize


def f(u, v):
    return jnp.exp(u @ v)


@pmp("in_axes", ((0, None), (1, None), (None, 0), (0, 0), (1, 1)))
@pmp("out_axes", (0, ))
def test_smap_f(in_axes, out_axes):
    fixed_shp = 3
    batched_shp = 4

    vf = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)
    sf = jft.smap(f, in_axes=in_axes, out_axes=out_axes)
    inp = []
    for i in in_axes:
        if i is None:
            shp = (fixed_shp, )
        elif i == 0:
            shp = (batched_shp, fixed_shp)
        else:
            assert i == 1
            shp = (fixed_shp, batched_shp)
        inp.append(jnp.ones(shp))
    assert_array_equal(vf(*inp), sf(*inp))


def g(u, v):
    return u, jnp.exp(u @ v)


@pmp("in_axes,out_axes", (((0, None), (0, 0)), ((None, 1), (None, 0))))
def test_smap_g(in_axes, out_axes):
    fixed_shp = 3
    batched_shp = 4

    vf = jax.vmap(g, in_axes=in_axes, out_axes=out_axes)
    sf = jft.smap(g, in_axes=in_axes, out_axes=out_axes)
    inp = []
    for i in in_axes:
        if i is None:
            shp = (fixed_shp, )
        elif i == 0:
            shp = (batched_shp, fixed_shp)
        else:
            assert i == 1
            shp = (fixed_shp, batched_shp)
        inp.append(jnp.ones(shp))
    v = vf(*inp)
    s = sf(*inp)
    jax.tree_map(assert_array_equal, v, s)
