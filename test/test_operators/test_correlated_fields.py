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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest
from numpy.testing import assert_allclose

import nifty7 as ift

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
@pmp('asperity', [None, (1, 1)])
@pmp('flexibility', [None, (1, 1)])
@pmp('ind', [None, 1])
@pmp('matern', [True, False])
def test_init(total_N, asperity, flexibility, ind, matern):
    if flexibility is None and asperity is not None:
        pytest.skip()
    cfg = 1, 1
    for dofdex in ([None], [None, [0]], [None, [0, 0], [0, 1], [1, 1]])[total_N]:
        cfm = ift.CorrelatedFieldMaker.make(0, cfg, '', total_N, dofdex)
        cfm.add_fluctuations(ift.RGSpace(4), cfg, flexibility, asperity, (-2, 0.1))
        if matern:
            if total_N == 0:
                cfm.add_fluctuations_matern(ift.RGSpace(4), *(3*[cfg]))
            else:
                with pytest.raises(NotImplementedError):
                    cfm.add_fluctuations_matern(ift.RGSpace(4), *(3*[cfg]))
        else:
            cfm.add_fluctuations(ift.RGSpace(4), *(4*[cfg]), index=ind)
        cfm.finalize(0)


@pmp('sspace', spaces)
@pmp('N', [0, 2])
def testAmplitudesInvariants(sspace, N):
    fsspace = ift.RGSpace((12,), (0.4,))
    dofdex1, dofdex2, dofdex3 = None, None, None
    if N == 2:
        dofdex1, dofdex2, dofdex3 = [0, 0], [1, 0], [1, 1]

    astds = 0.2, 1.2
    offset_std_mean = 1.3
    fa = ift.CorrelatedFieldMaker.make(1.2, (offset_std_mean, 1e-2), '', N,
                                       dofdex1)
    fa.add_fluctuations(sspace, (astds[0], 1e-2), (1.1, 2.), (2.1, .5), (-2, 1.),
                        'spatial', dofdex=dofdex2)
    fa.add_fluctuations(fsspace, (astds[1], 1e-2), (3.1, 1.), (.5, .1), (-4, 1.),
                        'freq', dofdex=dofdex3)
    op = fa.finalize()

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

    fa = ift.CorrelatedFieldMaker.make(0., (offset_std_mean, .1), '', N, dofdex1)
    fa.add_fluctuations(fsspace, (astds[1], 1.), (3.1, 1.), (.5, .1), (-4, 1.), 'freq',
                        dofdex=dofdex3)
    m = 3.
    x = fa.moment_slice_to_average(m)
    fa.add_fluctuations(sspace, (x, 1.5), (1.1, 2.), (2.1, .5), (-2, 1.), 'spatial', 0,
                        dofdex=dofdex2)
    op = fa.finalize()
    em, estd = _stats(fa.slice_fluctuation(0), samples)

    assert_allclose(m, em, rtol=0.5)
    assert op.target[-2] == sspace
    assert op.target[-1] == fsspace

    for ampl in fa.normalized_amplitudes:
        ift.extra.check_operator(ampl, 0.1*ift.from_random(ampl.domain), ntries=10)
    ift.extra.check_operator(op, 0.1*ift.from_random(op.domain), ntries=10)


@pmp('seed', [42, 31])
@pmp('domain', spaces)
@pmp('without', (('asperity', ), ('flexibility', ), ('flexibility', 'asperity')))
def test_complicated_vs_simple(seed, domain, without):
    with ift.random.Context(seed):
        offset_mean = _rand()
        offset_std = _posrand(), _posrand()
        fluctuations = _posrand(), _posrand()
        if "flexibility" in without:
            flexibility = None
        else:
            flexibility = _posrand(), _posrand()
        if "asperity" in without:
            asperity = None
        else:
            asperity = _posrand(), _posrand()
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
        cfm = ift.CorrelatedFieldMaker.make(offset_mean, offset_std, prefix)
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
        inp = ift.from_random(scf.domain)
        op1 = cfm.finalize()
        assert scf.domain is op1.domain
        ift.extra.assert_allclose(scf(inp), op1(inp))
        ift.extra.check_operator(scf, inp, ntries=10)

        op1 = cfm.amplitude
        op0 = scf.amplitude
        assert op0.domain is op1.domain
        ift.extra.assert_allclose(op0.force(inp), op1.force(inp))
