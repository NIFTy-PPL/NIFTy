#!/usr/bin/env python3
import numpy as np
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import pytest
import nifty8.re as jft
import nifty8 as ift
import jax.random as random
from numpy.testing import assert_allclose

pmp = pytest.mark.parametrize


@pmp('shape', [(2,), (3, 3)])
@pmp('distances', [.1])
@pmp('asperity', [None, (1., 1.)])
@pmp('flexibility', [None, (1., 1.)])
@pmp('fluctuations', [(1., 1.)])
@pmp('loglogavgslope', [(1., 1.)])
@pmp('matern', [False, True])
@pmp('scale', [(1., 1.)])
@pmp('loglogslope', [(1., 1.)])
@pmp('cutoff', [(1., 1.)])
def test_correlated_field_init(shape,
                               distances,
                               asperity,
                               flexibility,
                               fluctuations,
                               loglogavgslope,
                               matern,
                               scale,
                               loglogslope,
                               cutoff,
                               ):
    cf = jft.CorrelatedFieldMaker("cf")
    cf.set_amplitude_total_offset(offset_mean=0,
                                  offset_std=(.1, .1))
    if matern:
        cf.add_fluctuations_matern(
            shape,
            distances=distances,
            scale=scale,
            cutoff=cutoff,
            loglogslope=loglogslope,
        )
    else:
        cf.add_fluctuations(
            shape,
            distances=distances,
            asperity=asperity,
            flexibility=flexibility,
            fluctuations=fluctuations,
            loglogavgslope=loglogavgslope
        )
    correlated_field = cf.finalize()
    assert correlated_field


@pmp('seed', [0, 42])
@pmp('scale', [(1., 1.)])
@pmp('loglogslope', [(1., 1.)])
@pmp('cutoff', [(1., 1.)])
def test_re_matern_cf_vs_nifty(seed, scale, cutoff, loglogslope):
    shape = (3, 3)
    distances = 1./shape[0]
    rg_space = ift.RGSpace(shape, distances)
    offset_mean = 0.
    offset_std = (.1, .1)
    rg_key = random.PRNGKey(seed)
    mk_jcf = jft.CorrelatedFieldMaker("")
    mk_jcf.set_amplitude_total_offset(offset_mean=offset_mean,
                                      offset_std=offset_std)
    mk_jcf.add_fluctuations_matern(shape,
                                   distances=distances,
                                   scale=scale,
                                   cutoff=cutoff,
                                   loglogslope=loglogslope,
                                   non_parametric_kind="amplitude",
                                   renormalize_amplitude=False,
                                   )
    jcf = mk_jcf.finalize()

    mk_cf = ift.CorrelatedFieldMaker("")
    mk_cf.set_amplitude_total_offset(offset_mean, offset_std)
    mk_cf.add_fluctuations_matern(rg_space,
                                  scale=scale,
                                  cutoff=cutoff,
                                  loglogslope=loglogslope,
                                  )
    cf = mk_cf.finalize()

    random_pos = jft.random_like(rg_key, jcf.domain)
    nifty_random_pos = {key: ift.makeField(cf.domain[key], val)
                        for key, val in random_pos.items()}
    nifty_random_pos = ift.MultiField.from_dict(nifty_random_pos, cf.domain)
    assert_allclose(cf(nifty_random_pos).val, jcf(random_pos))


if __name__ == "__main__":
    test_correlated_field_init()
    test_re_matern_cf_vs_nifty()
