#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial

import jax
import pytest
from jax import numpy as jnp
from jax import random
from jax.tree_util import tree_map
from numpy.testing import assert_allclose

import nifty8.re as jft
from nifty8.re.likelihood import partial_insert_and_remove as jpartial

pmp = pytest.mark.parametrize


def _identity(x):
    return x


def tree_assert_allclose(a, b, **kwargs):
    return tree_map(partial(assert_allclose, **kwargs), a, b)


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


def _identity(x):
    return x


@pmp("seed", (33, 42, 43))
@pmp("forward", (_identity, jnp.exp))
def test_likelihood_partial(seed, forward):
    atol, rtol = 1e-14, 1e-14
    aallclose = partial(assert_allclose, rtol=rtol, atol=atol)

    key = random.PRNGKey(seed)
    forward = partial(tree_map, forward)

    domain = jft.Vector(
        {
            "a": jft.ShapeWithDtype(128),
            "b": jft.ShapeWithDtype(64)
        }
    )
    key, sk_d, sk_p = random.split(key, 3)
    primals = jft.random_like(sk_p, domain)
    data = forward(jft.random_like(sk_d, domain))

    gaussian = jft.Gaussian(data).amend(forward)

    gaussian_part, primals_liquid = gaussian.freeze(
        primals=primals, point_estimates=("b", )
    )
    assert primals_liquid.tree[0].shape == domain["a"].shape
    aallclose(gaussian_part(primals_liquid), gaussian(primals))
    tree_map(
        aallclose,
        gaussian_part.left_sqrt_metric(primals_liquid, data).tree[0],
        gaussian.left_sqrt_metric(primals, data).tree["a"],
    )
    rsm_orig = gaussian.right_sqrt_metric(primals, primals)
    rsm_orig.tree["b"] = jft.zeros_like(rsm_orig.tree["b"])
    tree_map(
        aallclose,
        gaussian_part.right_sqrt_metric(primals_liquid, primals_liquid),
        rsm_orig,
    )
    aallclose(
        gaussian_part.metric(primals_liquid, primals_liquid).tree[0],
        gaussian.metric(primals, primals).tree["a"],
    )


def test_likelihood_domain():
    lh = jft.Poissonian(jnp.zeros((3, 4), dtype=int))
    assert lh.domain is not None
    lh_wfwd = lh.amend(
        jft.Model(lambda x: x, domain=lh.domain, white_init=True)
    )
    assert lh_wfwd.domain is not None


def _Poissonian(x):
    return jft.Poissonian(tree_map(lambda y: jnp.exp(y).astype(int), x))


@pmp("seed", (33, 42, 43))
@pmp("likelihood", (jft.Gaussian, partial(jft.StudentT, dof=2.2), _Poissonian))
@pmp("forward_a,forward_b", ((_identity, jnp.exp), (jnp.exp, jnp.reciprocal)))
def test_nonvariable_likelihood_add(seed, likelihood, forward_a, forward_b):
    key = random.PRNGKey(seed)
    N_TRIES = 100
    shp_a, shp_b = (3, 5), (12, 5)
    swd_a = jft.Vector((jft.ShapeWithDtype(shp_a), ) * 2)
    swd_b = jft.Vector((jft.ShapeWithDtype(shp_b), ) * 3)
    key_a, key_b = "a", "b"

    def fwd_a(x):
        x = tree_map(forward_a, x[key_a])
        everything_pos = likelihood is _Poissonian
        return tree_map(jnp.abs, x) if everything_pos else x

    def fwd_b(x):
        x = tree_map(forward_b, x[key_b])
        everything_pos = likelihood is _Poissonian
        return tree_map(jnp.abs, x) if everything_pos else x

    def forward(x):
        return jft.Vector({key_a: fwd_a(x), key_b: fwd_b(x)})

    key, k_a, k_b = random.split(key, 3)
    data_a = jft.random_like(k_a, swd_a)
    data_b = jft.random_like(k_b, swd_b)
    data_ab = jft.Vector({key_a: data_a, key_b: data_b})

    swd = jft.Vector({key_a: swd_a, key_b: swd_b})
    lh_orig = likelihood(data_ab).amend(forward, domain=swd)
    lh_a = likelihood(data_a).amend(fwd_a, domain=jft.Vector({key_a: swd_a}))
    lh_b = likelihood(data_b).amend(fwd_b, domain=jft.Vector({key_b: swd_b}))
    lh_ab = lh_a + lh_b

    key, k_p, k_t, k_q = random.split(key, 4)
    rl = jax.vmap(jft.random_like, in_axes=(0, None))
    p, t, q = tuple(
        rl(random.split(k, N_TRIES), lh_orig.domain) for k in (k_p, k_t, k_q)
    )

    assert_allclose(jax.vmap(lh_orig)(p), jax.vmap(lh_ab)(p), equal_nan=False)
    rsm_orig = jax.vmap(lh_orig.right_sqrt_metric)(p, t)
    rsm_ab = jax.vmap(lh_ab.right_sqrt_metric)(p, t)
    tree_assert_allclose(
        rsm_orig.tree[key_a], rsm_ab["lh_left"], equal_nan=False
    )
    tree_assert_allclose(
        rsm_orig.tree[key_b], rsm_ab["lh_right"], equal_nan=False
    )
    tree_assert_allclose(
        jax.vmap(
            lambda p, t, q: lh_orig.
            left_sqrt_metric(p, lh_orig.right_sqrt_metric(t, q))
        )(p, t, q),
        jax.vmap(
            lambda p, t, q: lh_ab.
            left_sqrt_metric(p, lh_ab.right_sqrt_metric(t, q))
        )(p, t, q),
        equal_nan=False
    )
    tree_assert_allclose(
        jax.vmap(lh_orig.metric)(p, t),
        jax.vmap(lh_ab.metric)(p, t),
        equal_nan=False
    )
    nresi_orig = jax.vmap(lh_orig.normalized_residual)(p)
    nresi_ab = jax.vmap(lh_ab.normalized_residual)(p)
    tree_assert_allclose(
        nresi_orig.tree[key_a], nresi_ab["lh_left"], equal_nan=False
    )
    tree_assert_allclose(
        nresi_orig.tree[key_b], nresi_ab["lh_right"], equal_nan=False
    )
    trafo_orig = jax.vmap(lh_orig.transformation)(p)
    trafo_ab = jax.vmap(lh_ab.transformation)(p)
    tree_assert_allclose(
        trafo_orig.tree[key_a], trafo_ab["lh_left"], equal_nan=False
    )
    tree_assert_allclose(
        trafo_orig.tree[key_b], trafo_ab["lh_right"], equal_nan=False
    )


@pmp("seed", (33, 42, 43))
@pmp(
    "likelihood", (
        jft.VariableCovarianceGaussian,
        partial(jft.VariableCovarianceStudentT, dof=1.7),
    )
)
@pmp("forward_a,forward_b", ((_identity, jnp.exp), (jnp.exp, jnp.reciprocal)))
def test_variable_likelihood_add(seed, likelihood, forward_a, forward_b):
    key = random.PRNGKey(seed)
    N_TRIES = 100
    # Make input always a tuple for Variable* likelihoods
    data_shp_a = (3, 5)
    data_shp_b = (12, 5)
    primals_swd_a = (jft.ShapeWithDtype(data_shp_a), ) * 2
    data_swd_a = jft.ShapeWithDtype(data_shp_a)
    primals_swd_b = (jft.ShapeWithDtype(data_shp_b), ) * 2
    data_swd_b = jft.ShapeWithDtype(data_shp_b)
    key_a, key_b = "a", "b"

    def fwd_a(x):
        x1, x2 = tree_map(forward_a, x[key_a])
        return (x1, tree_map(jnp.abs, x2))

    def fwd_b(x):
        x1, x2 = tree_map(forward_b, x[key_b])
        return (x1, tree_map(jnp.abs, x2))

    def forward(x):
        a1, a2 = fwd_a(x)
        b1, b2 = fwd_b(x)
        x1 = jft.Vector({key_a: a1, key_b: b1})
        x2 = jft.Vector({key_a: a2, key_b: b2})
        return jft.Vector((x1, x2))

    key, k_a, k_b = random.split(key, 3)
    data_a = jft.random_like(k_a, data_swd_a)
    data_b = jft.random_like(k_b, data_swd_b)
    data_ab = jft.Vector({key_a: data_a, key_b: data_b})

    swd = jft.Vector({key_a: primals_swd_a, key_b: primals_swd_b})
    lh_orig = likelihood(data_ab).amend(forward, domain=swd)
    lh_a = likelihood(data_a).amend(
        fwd_a, domain=jft.Vector({key_a: primals_swd_a})
    )
    lh_b = likelihood(data_b).amend(
        fwd_b, domain=jft.Vector({key_b: primals_swd_b})
    )
    lh_ab = lh_a + lh_b

    key, k_p, k_t, k_q = random.split(key, 4)
    rl = jax.vmap(jft.random_like, in_axes=(0, None))
    p, t, q = tuple(
        rl(random.split(k, N_TRIES), lh_orig.domain) for k in (k_p, k_t, k_q)
    )

    assert_allclose(jax.vmap(lh_orig)(p), jax.vmap(lh_ab)(p), equal_nan=False)
    rsm_orig = jax.vmap(lh_orig.right_sqrt_metric)(p, t)
    rsm_ab = jax.vmap(lh_ab.right_sqrt_metric)(p, t)
    tree_assert_allclose(
        tuple(r.tree[key_a] for r in rsm_orig),
        rsm_ab["lh_left"],
        equal_nan=False
    )
    tree_assert_allclose(
        tuple(r.tree[key_b] for r in rsm_orig),
        rsm_ab["lh_right"],
        equal_nan=False
    )
    tree_assert_allclose(
        jax.vmap(
            lambda p, t, q: lh_orig.
            left_sqrt_metric(p, lh_orig.right_sqrt_metric(t, q))
        )(p, t, q),
        jax.vmap(
            lambda p, t, q: lh_ab.
            left_sqrt_metric(p, lh_ab.right_sqrt_metric(t, q))
        )(p, t, q),
        equal_nan=False
    )
    tree_assert_allclose(
        jax.vmap(lh_orig.metric)(p, t),
        jax.vmap(lh_ab.metric)(p, t),
        equal_nan=False
    )
    nresi_orig = lh_orig.normalized_residual(p)
    nresi_ab = lh_ab.normalized_residual(p)
    tree_assert_allclose(
        nresi_orig[key_a], nresi_ab["lh_left"], equal_nan=False
    )
    tree_assert_allclose(
        nresi_orig[key_b], nresi_ab["lh_right"], equal_nan=False
    )

    try:
        jax.vmap(lh_orig.transformation)(p)
    except NotImplementedError:
        return
    trafo_orig = jax.vmap(lh_orig.transformation)(p)
    trafo_ab = jax.vmap(lh_ab.transformation)(p)
    tree_assert_allclose(
        tuple(t[key_a] for t in trafo_orig),
        trafo_ab["lh_left"],
        equal_nan=False
    )
    tree_assert_allclose(
        tuple(t[key_b] for t in trafo_orig),
        trafo_ab["lh_right"],
        equal_nan=False
    )


if __name__ == "__main__":
    test_likelihood_partial(33, jnp.exp)
    test_nonvariable_likelihood_add(42, jft.Gaussian, jnp.exp, jnp.reciprocal)
    test_variable_likelihood_add(
        42, jft.VariableCovarianceGaussian, jnp.exp, jnp.reciprocal
    )
    test_nonvariable_likelihood_add(42, _Poissonian, jnp.exp, jnp.reciprocal)
