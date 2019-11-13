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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import pytest
from numpy.testing import assert_allclose
from numpy.random import seed

import nifty5 as ift


@pytest.mark.parametrize('sspace', [
    ift.RGSpace(4),
    ift.RGSpace((4, 4), (0.123,0.4)),
    ift.HPSpace(8),
    ift.GLSpace(4)
])
@pytest.mark.parametrize('rseed', [13, 2])
@pytest.mark.parametrize('Astds', [[1.,3.],[0.2,1.4]])
@pytest.mark.parametrize('offset_std', [1.,10.])
def testAmplitudesConsistency(rseed, sspace, Astds, offset_std):
    def stats(op,samples):
        sc = ift.StatCalculator()
        for s in samples:
            sc.add(op(s.extract(op.domain)))
        return sc.mean.to_global_data(), sc.var.sqrt().to_global_data()
    seed(rseed)
    nsam = 100

    fsspace = ift.RGSpace((12,), (0.4,))

    fa = ift.CorrelatedFieldMaker()
    fa.add_fluctuations(sspace, Astds[0], 1E-8, 1.1, 2., 2.1, .5,
                        -2, 1., 'spatial')
    fa.add_fluctuations(fsspace, Astds[1], 1E-8, 3.1, 1., .5, .1,
                        -4, 1., 'freq')
    op = fa.finalize(offset_std, 1E-8, '')

    samples = [ift.from_random('normal',op.domain) for _ in range(nsam)]
    tot_flm, _ = stats(fa.total_fluctuation,samples)
    offset_std,_ = stats(fa.amplitude_total_offset,samples)
    intergated_fluct_std0,_ = stats(fa.average_fluctuation(0),samples)
    intergated_fluct_std1,_ = stats(fa.average_fluctuation(1),samples)
    
    slice_fluct_std0,_ = stats(fa.slice_fluctuation(0),samples)
    slice_fluct_std1,_ = stats(fa.slice_fluctuation(1),samples)

    sams = [op(s) for s in samples]
    fluct_total = fa.total_fluctuation_realized(sams)
    fluct_space = fa.average_fluctuation_realized(sams,0)
    fluct_freq = fa.average_fluctuation_realized(sams,1)
    zm_std_mean = fa.offset_amplitude_realized(sams)
    sl_fluct_space = fa.slice_fluctuation_realized(sams,0)
    sl_fluct_freq = fa.slice_fluctuation_realized(sams,1)

    assert_allclose(offset_std, zm_std_mean, rtol=0.5)
    assert_allclose(intergated_fluct_std0, fluct_space, rtol=0.5)
    assert_allclose(intergated_fluct_std1, fluct_freq, rtol=0.5)
    assert_allclose(tot_flm, fluct_total, rtol=0.5)
    assert_allclose(slice_fluct_std0, sl_fluct_space, rtol=0.5)
    assert_allclose(slice_fluct_std1, sl_fluct_freq, rtol=0.5)

    fa = ift.CorrelatedFieldMaker()
    fa.add_fluctuations(fsspace, Astds[1], 1., 3.1, 1., .5, .1,
                        -4, 1., 'freq')
    m = 3.
    x = fa.moment_slice_to_average(m)
    fa.add_fluctuations(sspace, x, 1.5, 1.1, 2., 2.1, .5,
                        -2, 1., 'spatial', 0)
    op = fa.finalize(offset_std, .1, '')
    em, estd = stats(fa.slice_fluctuation(0),samples)

    assert_allclose(m, em, rtol=0.5)
    
    assert op.target[0] == sspace
    assert op.target[1] == fsspace