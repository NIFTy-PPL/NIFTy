#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import pytest

pytest.importorskip("jax")

import jax.random as random
import numpy as np
import pytest
from numpy.testing import assert_allclose

import nifty8 as ift
import nifty8.re as jft

pmp = pytest.mark.parametrize


@pmp('shape', [(2, ), (3, 3)])
@pmp('distances', [.1])
@pmp('offset_mean', [0])
@pmp('offset_std', [(.1, .1)])
@pmp('asperity', [None, (1., 1.)])
@pmp('flexibility', [None, (1., 1.)])
@pmp('fluctuations', [(1., 1.)])
@pmp('loglogavgslope', [(1., 1.)])
def test_correlated_field_non_parametric_init(
    shape,
    distances,
    offset_mean,
    offset_std,
    fluctuations,
    loglogavgslope,
    asperity,
    flexibility,
):
    cf = jft.CorrelatedFieldMaker("cf")
    cf.set_amplitude_total_offset(
        offset_mean=offset_mean, offset_std=offset_std
    )
    cf.add_fluctuations(
        shape,
        distances=distances,
        fluctuations=fluctuations,
        loglogavgslope=loglogavgslope,
        asperity=asperity,
        flexibility=flexibility,
    )
    correlated_field = cf.finalize()
    assert correlated_field
    assert correlated_field.domain


@pmp('shape', [(2, ), (3, 3)])
@pmp('distances', [.1])
@pmp('offset_mean', [0])
@pmp('offset_std', [(.1, .1)])
@pmp('scale', [(1., 1.)])
@pmp('loglogslope', [(1., 1.)])
@pmp('cutoff', [(1., 1.)])
def test_correlated_field_matern_init(
    shape,
    distances,
    offset_mean,
    offset_std,
    scale,
    loglogslope,
    cutoff,
):
    cf = jft.CorrelatedFieldMaker("cf")
    cf.set_amplitude_total_offset(
        offset_mean=offset_mean, offset_std=offset_std
    )
    cf.add_fluctuations_matern(
        shape,
        distances=distances,
        scale=scale,
        cutoff=cutoff,
        loglogslope=loglogslope,
        renormalize_amplitude=False,
    )
    correlated_field = cf.finalize()
    assert correlated_field
    assert correlated_field.domain


@pmp("flu", ([1e-1], [1e-1, 5e-3], [1e-1, 5e-3, 5e-3], 1e-1))
@pmp("slp", ([1e-1], [1e-1, 5e-3], [1e-1, 5e-3, 5e-3], 1e-1))
@pmp("flx", ([1e-1], [1e-1, 5e-3], [1e-1, 5e-3, 5e-3], 1e-1))
@pmp("asp", ([1e-1], [1e-1, 5e-3], [1e-1, 5e-3, 5e-3], 1e-1))
def test_correlated_field_non_parametric_init_validation(flu, slp, flx, asp):
    dims = (16, )
    if all(
        (
            isinstance(el, (tuple, list)) and len(el) == 2 and
            all(isinstance(f, float) for f in el)
        ) for el in (flu, slp, flx, asp)
    ):
        return
    with pytest.raises(TypeError):
        cf_zm = dict(offset_mean=0., offset_std=(1e-3, 1e-4))
        cf_fl = dict(
            fluctuations=flu,
            loglogavgslope=slp,
            flexibility=flx,
            asperity=asp,
        )
        cfm = jft.CorrelatedFieldMaker("cf")
        cfm.set_amplitude_total_offset(**cf_zm)
        cfm.add_fluctuations(
            shape=dims,
            distances=tuple(1. / d for d in dims),
            **cf_fl,
            prefix="ax1",
            non_parametric_kind="power"
        )
        cfm.finalize()


@pmp('seed', [0, 42])
@pmp('shape', [(4, ), (3, 3)])
@pmp('distances', [.1, 5.])
@pmp('offset_mean', [0])
@pmp('offset_std', [(.1, .1)])
@pmp('fluctuations', [(1., .1), (3., 2.)])
@pmp('loglogavgslope', [(-1., .1), (4., 1.)])
@pmp('flexibility', [(1., .1), (3., 2.)])
@pmp('asperity', [None, (.2, 2.e-2)])
def test_nifty_vs_niftyre_non_parametric_cf(
    seed, shape, distances, offset_mean, offset_std, fluctuations,
    loglogavgslope, asperity, flexibility
):
    key = random.PRNGKey(seed)
    fluct_kwargs = dict(
        fluctuations=fluctuations,
        loglogavgslope=loglogavgslope,
        asperity=asperity,
        flexibility=flexibility
    )

    jcfm = jft.CorrelatedFieldMaker("")
    jcfm.set_amplitude_total_offset(
        offset_mean=offset_mean, offset_std=offset_std
    )
    jcfm.add_fluctuations(
        shape,
        distances=distances,
        **fluct_kwargs,
        non_parametric_kind="power",
    )
    jcf = jcfm.finalize()

    cfm = ift.CorrelatedFieldMaker("")
    cfm.set_amplitude_total_offset(offset_mean, offset_std)
    cfm.add_fluctuations(ift.RGSpace(shape, distances), **fluct_kwargs)
    cf = cfm.finalize(prior_info=0)

    pos = jft.random_like(key, jcf.domain)
    npos = {k: ift.makeField(cf.domain[k], v if k != "spectrum" else v.T) for
            k, v in pos.items()}
    npos = ift.MultiField.from_dict(npos, cf.domain)
    assert_allclose(cf(npos).val, jcf(pos))


@pmp('seed', [0, 42])
@pmp('shape', [(2, ), (3, 3)])
@pmp('distances', [.1, 5.])
@pmp('offset_mean', [0])
@pmp('offset_std', [(.1, .1)])
@pmp('scale', [(1., 1.), (3., 2.)])
@pmp('loglogslope', [(1., 1.), (5., 0.5)])
@pmp('cutoff', [(1., 1.), (0.1, 0.01)])
def test_nifty_vs_niftyre_matern_cf(
    seed, shape, distances, offset_mean, offset_std, scale, cutoff, loglogslope
):
    key = random.PRNGKey(seed)
    fluct_kwargs = dict(scale=scale, cutoff=cutoff, loglogslope=loglogslope)

    jcfm = jft.CorrelatedFieldMaker("")
    jcfm.set_amplitude_total_offset(
        offset_mean=offset_mean, offset_std=offset_std
    )
    jcfm.add_fluctuations_matern(
        shape,
        distances=distances,
        **fluct_kwargs,
        non_parametric_kind="amplitude",
        renormalize_amplitude=False,
    )
    jcf = jcfm.finalize()

    cfm = ift.CorrelatedFieldMaker("")
    cfm.set_amplitude_total_offset(offset_mean, offset_std)
    cfm.add_fluctuations_matern(ift.RGSpace(shape, distances), **fluct_kwargs)
    cf = cfm.finalize(prior_info=0)

    pos = jft.random_like(key, jcf.domain)
    npos = {k: ift.makeField(cf.domain[k], v) for k, v in pos.items()}
    npos = ift.MultiField.from_dict(npos, cf.domain)
    assert_allclose(cf(npos).val, jcf(pos))
