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

import nifty8 as ift
import numpy as np

Nsamp = 20000
np.random.seed(42)


def _to_array(d):
    if isinstance(d, np.ndarray):
        return d
    assert isinstance(d, dict)
    return np.concatenate(list(d.values()))


def test_GaussianEnergy():
    sp = ift.UnstructuredDomain(Nsamp)
    S = ift.ScalingOperator(sp, 1., complex)
    samp = S.draw_sample()
    real_std = np.std(samp.val.real)
    imag_std = np.std(samp.val.imag)
    np.testing.assert_allclose(real_std, imag_std,
                               atol=5./np.sqrt(Nsamp))
    sp1 = ift.UnstructuredDomain(1)
    mean = ift.full(sp1, 0.)
    real_icov = ift.ScalingOperator(sp1, real_std**(-2))
    imag_icov = ift.ScalingOperator(sp1, imag_std**(-2))
    real_energy = ift.GaussianEnergy(mean, inverse_covariance=real_icov)
    imag_energy = ift.GaussianEnergy(mean, inverse_covariance=imag_icov)
    icov = ift.ScalingOperator(sp1, 1.)
    complex_energy = ift.GaussianEnergy(mean+0.j, inverse_covariance=icov)
    for i in range(min(10, Nsamp)):
        fld = ift.full(sp1, samp.val[i])
        val1 = (real_energy(fld.real) + imag_energy(fld.imag)).val
        val2 = complex_energy(fld).val
        np.testing.assert_allclose(val1, val2, atol=10./np.sqrt(Nsamp))
