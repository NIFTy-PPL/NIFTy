#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np
import pytest

from nifty8.re import refine, refine_util

pmp = pytest.mark.parametrize


def matern_kernel(distance, scale, cutoff, dof):
    from jax.scipy.special import gammaln
    from scipy.special import kv

    reg_dist = jnp.sqrt(2 * dof) * distance / cutoff
    return scale**2 * 2**(1 - dof) / jnp.exp(
        gammaln(dof)
    ) * (reg_dist)**dof * kv(dof, reg_dist)


scale, cutoff, dof = 1., 80., 3 / 2

x = jnp.logspace(-6, 11, base=jnp.e, num=int(1e+5))
y = matern_kernel(x, scale, cutoff, dof)
y = jnp.nan_to_num(y, nan=0.)
kernel = Partial(jnp.interp, xp=x, fp=y)


@pmp("shape0", ((16, ), (13, 15), (11, 12, 13)))
@pmp("depth", (1, 2))
@pmp("_coarse_size", (3, 5, 7))
@pmp("_fine_size", (2, 4, 6))
@pmp("_fine_strategy", ("jump", "extend"))
def test_shape_translations(
    shape0, depth, _coarse_size, _fine_size, _fine_strategy
):
    kwargs = {
        "_coarse_size": _coarse_size,
        "_fine_size": _fine_size,
        "_fine_strategy": _fine_strategy
    }

    eval_cf_shape = partial(
        jax.eval_shape,
        partial(
            refine.correlated_field,
            distances0=(1., ) * len(shape0),
            kernel=lambda x: x,
            **kwargs
        )
    )
    dom = refine.get_refinement_shapewithdtype(shape0, depth, **kwargs)
    tgt = eval_cf_shape(dom)
    tgt_pred_shp = refine_util.coarse2fine_shape(shape0, depth, **kwargs)
    assert tgt_pred_shp == tgt.shape
    assert dom[-1].size == tgt.size == np.prod(tgt_pred_shp)

    shape0_pred = refine_util.fine2coarse_shape(tgt.shape, depth, **kwargs)
    dom_pred = refine.get_refinement_shapewithdtype(
        shape0_pred, depth, **kwargs
    )
    tgt_pred = eval_cf_shape(dom_pred)

    assert tgt.shape == tgt_pred.shape
    if _fine_strategy == "jump":
        assert shape0_pred == shape0
    else:
        assert _fine_strategy == "extend"
        assert all(s0_p <= s0 for s0_p, s0 in zip(shape0_pred, shape0))


@pmp("seed", (42, 45))
def test_gauss_kl(seed, n_resamples=100):
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        d = max(rng.poisson(4), 1)
        m_t = rng.normal(size=(d, d))
        m_t = m_t @ m_t.T
        scl = rng.lognormal(2., 3.)

        np.testing.assert_allclose(
            refine_util.gauss_kl(m_t, m_t), 0., atol=1e-11
        )
        kl_rhs_scl = 0.5 * d * (np.log(scl) + 1. / scl - 1.)
        np.testing.assert_allclose(
            kl_rhs_scl, refine_util.gauss_kl(m_t, scl * m_t), rtol=1e-11
        )
        kl_lhs_scl = 0.5 * d * (-np.log(scl) + scl - 1.)
        np.testing.assert_allclose(
            kl_lhs_scl, refine_util.gauss_kl(scl * m_t, m_t), rtol=1e-10
        )
