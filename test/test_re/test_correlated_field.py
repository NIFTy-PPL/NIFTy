#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause

import pytest

pytest.importorskip("jax")

import jax.random as random
import pytest
from numpy.testing import assert_allclose

import nifty8 as ift
import nifty8.re as jft

pmp = pytest.mark.parametrize


@pmp("shape", [(2,), (3, 3)])
@pmp("distances", [0.1])
@pmp("offset_mean", [0])
@pmp("offset_std", [(0.1, 0.1)])
@pmp("asperity", [None, (1.0, 1.0)])
@pmp("flexibility", [None, (1.0, 1.0)])
@pmp("fluctuations", [(1.0, 1.0)])
@pmp("loglogavgslope", [(1.0, 1.0)])
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
    cf.set_amplitude_total_offset(offset_mean=offset_mean, offset_std=offset_std)
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


@pmp("shape", [(2,), (3, 3)])
@pmp("distances", [0.1])
@pmp("offset_mean", [0])
@pmp("offset_std", [(0.1, 0.1)])
@pmp("scale", [(1.0, 1.0)])
@pmp("loglogslope", [(1.0, 1.0)])
@pmp("cutoff", [(1.0, 1.0)])
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
    cf.set_amplitude_total_offset(offset_mean=offset_mean, offset_std=offset_std)
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
    dims = (16,)
    if all(
        (
            isinstance(el, (tuple, list))
            and len(el) == 2
            and all(isinstance(f, float) for f in el)
        )
        for el in (flu, slp, flx, asp)
    ):
        return
    with pytest.raises(TypeError):
        cf_zm = dict(offset_mean=0.0, offset_std=(1e-3, 1e-4))
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
            distances=tuple(1.0 / d for d in dims),
            **cf_fl,
            prefix="ax1",
            non_parametric_kind="power",
        )
        cfm.finalize()


@pmp("seed", [0, 42])
@pmp("ht_convention", ["canonical_hartley", "non_canonical_hartley"])
@pmp(
    "shape,distances,harmonic_type",
    [
        ((4,), 0.1, "fourier"),
        ((4,), 5.0, "fourier"),
        ((3, 3), 0.1, "fourier"),
        ((3, 3), 5.0, "fourier"),
        ((4,), None, "spherical"),
        ((32,), None, "spherical"),
    ],
)
@pmp("offset_mean", [0])
@pmp("offset_std", [(0.1, 0.1)])
@pmp("fluctuations", [(1.0, 0.1), (3.0, 2.0)])
@pmp("loglogavgslope", [(-1.0, 0.1), (4.0, 1.0)])
@pmp("flexibility", [(1.0, 0.1), (3.0, 2.0)])
@pmp("asperity", [None, (0.2, 2.0e-2)])
def test_nifty_vs_niftyre_non_parametric_cf(
    seed,
    ht_convention,
    shape,
    distances,
    harmonic_type,
    offset_mean,
    offset_std,
    fluctuations,
    loglogavgslope,
    asperity,
    flexibility,
):
    if harmonic_type == "spherical":
        try:
            from jaxbind.contrib import jaxducc0
        except (ImportError, ModuleNotFoundError):
            pytest.skip("skipping since `jaxducc0` from `jaxbind` is not available")
    jft.config.update("hartley_convention", ht_convention)

    key = random.PRNGKey(seed)
    fluct_kwargs = dict(
        fluctuations=fluctuations,
        loglogavgslope=loglogavgslope,
        asperity=asperity,
        flexibility=flexibility,
    )

    jcfm = jft.CorrelatedFieldMaker("")
    jcfm.set_amplitude_total_offset(offset_mean=offset_mean, offset_std=offset_std)
    jcfm.add_fluctuations(
        shape,
        distances=distances,
        **fluct_kwargs,
        non_parametric_kind="power",
        harmonic_type=harmonic_type,
    )
    jcf = jcfm.finalize()

    cfm = ift.CorrelatedFieldMaker("")
    cfm.set_amplitude_total_offset(offset_mean, offset_std)
    sp = (
        ift.RGSpace(shape, distances)
        if distances is not None
        else ift.HPSpace(shape[0])
    )
    cfm.add_fluctuations(sp, **fluct_kwargs)
    cf = cfm.finalize(prior_info=0)

    pos = jft.random_like(key, jcf.domain)
    npos = {
        k: ift.makeField(cf.domain[k], v if k != "spectrum" else v.T)
        for k, v in pos.items()
    }
    npos = ift.MultiField.from_dict(npos, cf.domain)
    assert_allclose(cf(npos).val, jcf(pos))


@pmp("seed", [0, 42])
@pmp("shape", [(2,), (3, 3)])
@pmp("distances", [0.1, 5.0])
@pmp("offset_mean", [0])
@pmp("offset_std", [(0.1, 0.1)])
@pmp("scale", [(1.0, 1.0), (3.0, 2.0)])
@pmp("loglogslope", [(1.0, 1.0), (5.0, 0.5)])
@pmp("cutoff", [(1.0, 1.0), (0.1, 0.01)])
def test_nifty_vs_niftyre_matern_cf(
    seed, shape, distances, offset_mean, offset_std, scale, cutoff, loglogslope
):
    key = random.PRNGKey(seed)
    fluct_kwargs = dict(scale=scale, cutoff=cutoff, loglogslope=loglogslope)

    jcfm = jft.CorrelatedFieldMaker("")
    jcfm.set_amplitude_total_offset(offset_mean=offset_mean, offset_std=offset_std)
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


CFG_OFFSET = dict(offset_mean=0, offset_std=(0.1, 0.1))
CFG_FLUCT = dict(
    fluctuations=(1.0, 0.1),
    loglogavgslope=(-1.0, 0.1),
    flexibility=(1.0, 0.1),
    asperity=(0.2, 2.0e-2),
)
FLUCTUATIONS_CHOICES = (
    dict(shape=(4,), distances=None, harmonic_type="spherical"),
    dict(shape=(3, 3), distances=(0.1, 0.1), harmonic_type="fourier"),
    dict(shape=(6,), distances=(1.0,), harmonic_type="fourier"),
)


@pmp("seed", [0, 42])
@pmp("fluct1", FLUCTUATIONS_CHOICES)
@pmp("fluct2", FLUCTUATIONS_CHOICES)
def test_nifty_vs_niftyre_product(seed, fluct1, fluct2):
    if any(f["harmonic_type"] == "spherical" for f in (fluct1, fluct2)):
        try:
            from jaxbind.contrib import jaxducc0
        except (ImportError, ModuleNotFoundError):
            pytest.skip("skipping since `jaxducc0` from `jaxbind` is not available")
    key = random.PRNGKey(seed)

    jcfm = jft.CorrelatedFieldMaker("")
    jcfm.set_amplitude_total_offset(**CFG_OFFSET)
    for i, f in enumerate((fluct1, fluct2)):
        jcfm.add_fluctuations(
            **f, **CFG_FLUCT, non_parametric_kind="power", prefix=f"space{i}"
        )
    jcf = jcfm.finalize()

    cfm = ift.CorrelatedFieldMaker("")
    cfm.set_amplitude_total_offset(**CFG_OFFSET)
    for i, f in enumerate((fluct1, fluct2)):
        sp = (
            ift.HPSpace(f["shape"][0])
            if f["distances"] is None
            else ift.RGSpace(f["shape"], f["distances"])
        )
        cfm.add_fluctuations(sp, **CFG_FLUCT, prefix=f"space{i}")
    cf = cfm.finalize(prior_info=0)

    pos = jft.random_like(key, jcf.domain)
    npos = {
        k: ift.makeField(cf.domain[k], v if not k.endswith("spectrum") else v.T)
        for k, v in pos.items()
    }
    npos = ift.MultiField.from_dict(npos, cf.domain)
    assert_allclose(cf(npos).val, jcf(pos))
