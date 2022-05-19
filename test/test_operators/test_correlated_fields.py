# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..common import setup_function, teardown_function

pmp = pytest.mark.parametrize
spaces = [
    ift.RGSpace(4),
    ift.RGSpace((4, 4), (0.123, 0.4)),
    ift.HPSpace(8),
    ift.GLSpace(4)
]


def _stats(op, samples):
    sc = ift.StatCalculator()
    for s in samples:
        sc.add(op(s.extract(op.domain)))
    return sc.mean.val, sc.var.ptw("sqrt").val


def _rand():
    return ift.random.current_rng().normal()


def _posrand():
    return np.exp(_rand())


@pmp('dofdex', [[0, 0], [0, 1]])
@pmp('seed', [12, 3])
def testDistributor(dofdex, seed):
    with ift.random.Context(seed):
        dom = ift.RGSpace(3)
        N_copies = max(dofdex) + 1
        distributed_target = ift.makeDomain(
            (ift.UnstructuredDomain(len(dofdex)), dom))
        target = ift.makeDomain((ift.UnstructuredDomain(N_copies), dom))
        op = ift.library.correlated_fields._Distributor(
            dofdex, target, distributed_target)
        ift.extra.check_linear_operator(op)


@pmp('total_N', [0, 1, 2])
@pmp('offset_std', [None, (1, 1)])
@pmp('asperity', [None, (1, 1)])
@pmp('flexibility', [None, (1, 1)])
@pmp('ind', [None, 1])
@pmp('matern', [True, False])
def test_init(total_N, offset_std, asperity, flexibility, ind, matern):
    if flexibility is None and asperity is not None:
        pytest.skip()
    cfg = 1, 1
    for dofdex in ([None], [None, [0]], [None, [0, 0], [0, 1], [1, 1]])[total_N]:
        cfm = ift.CorrelatedFieldMaker('', total_N)
        cfm.add_fluctuations(ift.RGSpace(4), cfg, flexibility, asperity, (-2, 0.1))
        if not offset_std is None:
            if matern:
                if total_N == 0:
                    cfm.add_fluctuations_matern(ift.RGSpace(4), *(3*[cfg]))
                else:
                    with pytest.raises(NotImplementedError):
                        cfm.add_fluctuations_matern(ift.RGSpace(4), *(3*[cfg]))
            else:
                cfm.add_fluctuations(ift.RGSpace(4), *(4*[cfg]), index=ind)
        cfm.set_amplitude_total_offset(0, offset_std, dofdex=dofdex)
        cfm.finalize(prior_info=0)


@pmp('sspace', spaces)
@pmp('asperity', [None, (1, 1)])
@pmp('flexibility', [None, (1, 1)])
@pmp('matern', [True, False])
def test_unit_zero_mode(sspace, asperity, flexibility, matern):
    if flexibility is None and asperity is not None:
        pytest.skip()
    cfg = 1, 1
    cfm = ift.CorrelatedFieldMaker('')
    if matern:
        cfm.add_fluctuations_matern(sspace, *(3 * [cfg]))
    else:
        cfm.add_fluctuations(sspace, *(4 * [cfg]))
    cfm.set_amplitude_total_offset(0, 1.)
    cf = cfm.finalize(prior_info=0)

    r = ift.from_random(cf.domain).to_dict()
    r_xi = np.copy(r["xi"].val)
    r_xi[0] = 1.
    r["xi"] = ift.Field(r["xi"].domain, r_xi)
    r = ift.MultiField.from_dict(r)

    cf_r = cf(r)
    rtol = 1e-7
    if isinstance(sspace, (ift.HPSpace, ift.GLSpace)):
        rtol = 1e-2
    assert_allclose(cf_r.s_integrate(), sspace.total_volume, rtol=rtol)

@pmp('sspace', spaces)
@pmp('asperity', [None, (1, 1)])
@pmp('flexibility', [None, (1, 1)])
@pmp('matern', [True, False])
def test_constant_zero_mode(sspace, asperity, flexibility, matern):
    if flexibility is None and asperity is not None:
        pytest.skip()
    cfg = 1, 1
    cfm = ift.CorrelatedFieldMaker('')
    if matern:
        cfm.add_fluctuations_matern(sspace, *(3 * [cfg]))
    else:
        fl = (1., 0.5)
        ll = (-4, 1)
        cfm.add_fluctuations(sspace, fl, flexibility, asperity, ll)
        
        cf_simple = ift.SimpleCorrelatedField(sspace, 0, None, fl, flexibility, asperity, ll)
    cfm.set_amplitude_total_offset(0, None)
    cf = cfm.finalize(prior_info=0)

    r = ift.from_random(cf.domain)
    cf_r = cf(r)
    atol = 1e-8
    if isinstance(sspace, (ift.HPSpace, ift.GLSpace)):
        atol = 1e-2
        if matern:
            atol = 1e-1
    assert_allclose(cf_r.s_integrate(), 0., atol=atol)
    if not matern:
        cf_r_simple = cf_simple(r)
        assert_allclose(cf_r_simple.s_integrate(), 0., atol=atol)

@pmp('sspace', spaces)
@pmp('N', [0, 2])
def testAmplitudesInvariants(sspace, N):
    fsspace = ift.RGSpace((12,), (0.4,))
    dofdex1, dofdex2, dofdex3 = None, None, None
    if N == 2:
        dofdex1, dofdex2, dofdex3 = [0, 0], [1, 0], [1, 1]

    astds = 0.2, 1.2
    offset_std_mean = 1.3
    fa = ift.CorrelatedFieldMaker('', N)
    fa.add_fluctuations(sspace, (astds[0], 1e-2), (1.1, 2.), (2.1, .5), (-2, 1.),
                        'spatial', dofdex=dofdex2)
    fa.add_fluctuations(fsspace, (astds[1], 1e-2), (3.1, 1.), (.5, .1), (-4, 1.),
                        'freq', dofdex=dofdex3)
    fa.set_amplitude_total_offset(1.2, (offset_std_mean, 1e-2), dofdex=dofdex1)
    op = fa.finalize(prior_info=0)

    samples = [ift.from_random(op.domain) for _ in range(100)]
    tot_flm, _ = _stats(fa.total_fluctuation, samples)
    offset_amp_std, _ = _stats(fa.amplitude_total_offset, samples)
    intergated_fluct_std0, _ = _stats(fa.average_fluctuation(0), samples)
    intergated_fluct_std1, _ = _stats(fa.average_fluctuation(1), samples)

    slice_fluct_std0, _ = _stats(fa.slice_fluctuation(0), samples)
    slice_fluct_std1, _ = _stats(fa.slice_fluctuation(1), samples)

    sams = [op(s) for s in samples]
    fluct_total = fa.total_fluctuation_realized(sams)
    fluct_space = fa.average_fluctuation_realized(sams, 0)
    fluct_freq = fa.average_fluctuation_realized(sams, 1)
    zm_std_mean = fa.offset_amplitude_realized(sams)
    sl_fluct_space = fa.slice_fluctuation_realized(sams, 0)
    sl_fluct_freq = fa.slice_fluctuation_realized(sams, 1)

    assert_allclose(offset_amp_std, zm_std_mean, rtol=0.5)
    assert_allclose(intergated_fluct_std0, fluct_space, rtol=0.5)
    assert_allclose(intergated_fluct_std1, fluct_freq, rtol=0.5)
    assert_allclose(tot_flm, fluct_total, rtol=0.5)
    assert_allclose(slice_fluct_std0, sl_fluct_space, rtol=0.5)
    assert_allclose(slice_fluct_std1, sl_fluct_freq, rtol=0.5)

    fa = ift.CorrelatedFieldMaker('', N)
    fa.set_amplitude_total_offset(0., (offset_std_mean, .1), dofdex=dofdex1)
    fa.add_fluctuations(fsspace, (astds[1], 1.), (3.1, 1.), (.5, .1), (-4, 1.), 'freq',
                        dofdex=dofdex3)
    m = 3.
    x = fa.moment_slice_to_average(m)
    fa.add_fluctuations(sspace, (x, 1.5), (1.1, 2.), (2.1, .5), (-2, 1.), 'spatial', 0,
                        dofdex=dofdex2)
    op = fa.finalize(prior_info=0)
    em, estd = _stats(fa.slice_fluctuation(0), samples)

    assert_allclose(m, em, rtol=0.5)
    assert op.target[-2] == sspace
    assert op.target[-1] == fsspace

    for ampl in fa.get_normalized_amplitudes():
        ift.extra.check_operator(ampl, 0.1*ift.from_random(ampl.domain), ntries=10)
    ift.extra.check_operator(op, 0.1*ift.from_random(op.domain), ntries=10)


@pmp('seed', [42, 31])
@pmp('domain', spaces)
@pmp('without', (('offset_std', ), ('asperity', ), ('flexibility', ), ('flexibility', 'asperity')))
def test_complicated_vs_simple(seed, domain, without):
    with ift.random.Context(seed):
        offset_mean = _rand()
        fluctuations = _posrand(), _posrand()
        if "flexibility" in without:
            flexibility = None
        else:
            flexibility = _posrand(), _posrand()
        if "asperity" in without:
            asperity = None
        else:
            asperity = _posrand(), _posrand()
        if "offset_std" in without:
            offset_std = None
        else:
            offset_std = _posrand(), _posrand()
        loglogavgslope = _posrand(), _posrand()
        prefix = 'foobar'
        hspace = domain.get_default_codomain()
        scf_args = (
            domain,
            offset_mean,
            offset_std,
            fluctuations,
            flexibility,
            asperity,
            loglogavgslope
        )
        add_fluct_args = (
            domain,
            fluctuations,
            flexibility,
            asperity,
            loglogavgslope
        )
        cfm = ift.CorrelatedFieldMaker(prefix)
        if asperity is not None and flexibility is None:
            with pytest.raises(ValueError):
                scf = ift.SimpleCorrelatedField(*scf_args, prefix=prefix,
                                                harmonic_partner=hspace)
            with pytest.raises(ValueError):
                cfm.add_fluctuations(*add_fluct_args, prefix='',
                                     harmonic_partner=hspace)
            return
        scf = ift.SimpleCorrelatedField(*scf_args, prefix=prefix,
                                        harmonic_partner=hspace)
        cfm.add_fluctuations(*add_fluct_args, prefix='',
                             harmonic_partner=hspace)
        cfm.set_amplitude_total_offset(offset_mean, offset_std)
        inp = ift.from_random(scf.domain)
        op1 = cfm.finalize(prior_info=0)
        assert scf.domain is op1.domain
        ift.extra.assert_allclose(scf(inp), op1(inp))
        ift.extra.check_operator(scf, inp, ntries=10)

        op1 = cfm.amplitude
        op0 = scf.amplitude
        assert op0.domain is op1.domain
        ift.extra.assert_allclose(op0.force(inp), op1.force(inp))
