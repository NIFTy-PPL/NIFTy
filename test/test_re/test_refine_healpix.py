#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import sys
from functools import partial

import pytest

pytest.importorskip("jax")
pytest.importorskip("healpy")

import jax
from jax import random
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import nifty8.re as jft

pmp = pytest.mark.parametrize

jax.config.update("jax_enable_x64", True)


def lst2fixt(lst):
    @pytest.fixture(params=lst)
    def fixt(request):
        return request.param

    return fixt


def matern_kernel(distance, scale, cutoff, dof):
    from jax.scipy.special import gammaln
    from scipy.special import kv

    reg_dist = jnp.sqrt(2 * dof) * distance / cutoff
    return scale**2 * 2**(1 - dof) / jnp.exp(
        gammaln(dof)
    ) * (reg_dist)**dof * kv(dof, reg_dist)


scale, cutoff, dof = 1., 2., 3 / 2

x = jnp.logspace(-6, 11, base=jnp.e, num=int(1e+5))
y = matern_kernel(x, scale, cutoff, dof)
y = jnp.nan_to_num(y, nan=0.)
kernel = Partial(jnp.interp, xp=x, fp=y)
inv_kernel = Partial(jnp.interp, xp=y, fp=x)

cc_0 = jft.HEALPixChart(
    min_shape=(
        12 * 2**2,
        6,
    ),
    nonhp_rg2cart=lambda x: (jnp.exp(0.3 * x[0] - 0.3), ),
    nonhp_cart2rg=lambda x: ((jnp.log(x[0]) + 0.3) / 0.3, ),
    depth=1,
    _coarse_size=3,
    _fine_size=2,
)
cc_1 = jft.HEALPixChart(
    min_shape=(
        12 * 4**2,
        6,
    ),
    nonhp_rg2cart=lambda x: (jnp.exp(0.75 * x[0] - 1.), ),
    nonhp_cart2rg=lambda x: ((jnp.log(x[0]) + 1.) / 0.75, ),
    depth=1,
    _coarse_size=3,
    _fine_size=2,
)
cc = lst2fixt((cc_0, cc_1))


@pmp(
    "atol,rtol,which", (
        (1e-2 * scale, 5e-2 * scale, "cov"),
        (1e-3 * scale, 5e-3 * scale, "cov"),
        (1e-4 * scale, 5e-2 * scale, "dist")
    )
)
def test_healpix_refinement_matrices_uniquifying(
    cc, atol, rtol, which, kernel=kernel
):
    seed = 42
    aassert = partial(
        assert_allclose,
        atol=1e-11 * scale,
        rtol=1e-10 * scale,
        equal_nan=False
    )

    rf = jft.RefinementHPField(cc, kernel=kernel)
    rfm = rf.matrices()
    mbs = 12 * cc.nside**2 - 1
    rfm_u = rf.matrices(atol=atol, rtol=rtol, which=which, mat_buffer_size=mbs)
    assert rfm_u.index_map[-1].size == cc.shape_at(cc.depth - 1)[0]
    # Assert that we are actually using fewer matrices
    assert np.unique(rfm_u.index_map[-1]).size < rfm_u.index_map[-1].size
    msg = f"{np.unique(rfm_u.index_map[-1]).size} !< {cc.shape_at(cc.depth - 1)[0]}"
    print(msg, file=sys.stderr)
    assert_array_equal(rfm_u.cov_sqrt0, rfm.cov_sqrt0)
    for lvl in range(cc.depth):
        naive_f, naive_p = rfm.filter[lvl], rfm.propagator_sqrt[lvl]
        uniq_f = rfm_u.filter[lvl][rfm_u.index_map[lvl]]
        uniq_p = rfm_u.propagator_sqrt[lvl][rfm_u.index_map[lvl]]
        aassert(uniq_f, naive_f, rtol=1e-7)
        aassert(uniq_p, naive_p, rtol=1e-7)

    key = random.PRNGKey(seed)
    key, sk = random.split(key)
    exc = jft.random_like(sk, rf.domain)
    f = rf(exc, kernel=rfm)
    f_u = rf(exc, kernel=rfm_u)
    aassert(f_u, f)


@pmp("atol,rtol,which", ((1e-2 * scale, 5e-2 * scale, "cov"), (None, ) * 3))
def test_healpix_refinement(cc, atol, rtol, which, kernel=kernel):
    cov_sqrt = jft.refine.healpix_refine.cov_sqrt(cc, kernel, cc.depth)
    cov = cov_sqrt @ cov_sqrt.T

    cf = jft.RefinementHPField(cc, kernel)
    mbs = 12 * cc.nside**2 - 1
    rfm = cf.matrices(atol=atol, rtol=rtol, which=which, mat_buffer_size=mbs)
    if which is not None:
        assert rfm.index_map is not None
        # Assert that we are actually using fewer matrices
        assert np.unique(rfm.index_map[-1]).size < rfm.index_map[-1].size
    cf_wm = jft.Model(partial(cf, kernel=rfm), domain=cf.domain)
    cov_refinement = jft.refine.util.refinement_covariance(cf_wm, kernel)

    assert_allclose(
        cov_refinement,
        cov,
        atol=0.2 * scale,
        rtol=0.1 * scale,
        equal_nan=False
    )


if __name__ == "__main__":
    test_healpix_refinement_matrices_uniquifying(
        cc_0, atol=1e-2, rtol=1e-5, which="cov", kernel=kernel
    )
    test_healpix_refinement(
        cc_0, atol=1e-2, rtol=5e-2, which="cov", kernel=kernel
    )
