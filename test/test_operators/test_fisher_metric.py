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

import nifty6 as ift

from ..common import list2fixture, setup_function, teardown_function

spaces = [ift.GLSpace(5),
          ift.MultiDomain.make({'': ift.RGSpace(5, distances=.789)}),
          (ift.RGSpace(3, distances=.789), ift.UnstructuredDomain(2))]
pmp = pytest.mark.parametrize
field = list2fixture([ift.from_random(sp, 'normal') for sp in spaces] +
        [ift.from_random(sp, 'normal', dtype=np.complex128) for sp in spaces])

Nsamp = 2000
np.random.seed(42)

def _to_array(d):
    if isinstance(d, np.ndarray):
        return d
    assert isinstance(d, dict)
    return np.concatenate(list(d.values()))

def energy_tester(pos, get_noisy_data, energy_initializer):
    domain = pos.domain
    test_vec = ift.from_random(domain, 'normal')
    results = []
    lin = ift.Linearization.make_var(pos)
    for i in range(Nsamp):
        data = get_noisy_data(pos)
        energy = energy_initializer(data)
        grad = energy(lin).jac.adjoint(ift.full(energy.target, 1.))
        results.append(_to_array((grad*grad.s_vdot(test_vec)).val))
    res = np.mean(np.array(results), axis=0)
    std = np.std(np.array(results), axis=0)/np.sqrt(Nsamp)
    energy = energy_initializer(data)
    lin = ift.Linearization.make_var(pos, want_metric=True)
    res2 = _to_array(energy(lin).metric(test_vec).val)
    np.testing.assert_allclose(res/std, res2/std, atol=6)

def test_GaussianEnergy(field):
    dtype = field.dtype
    icov = ift.from_random(field.domain, 'normal')**2
    icov = ift.makeOp(icov)
    get_noisy_data = lambda mean : mean+icov.draw_sample_with_dtype(
            from_inverse=True, dtype=dtype)
    E_init = lambda mean : ift.GaussianEnergy(mean=mean,
            inverse_covariance=icov)
    energy_tester(field, get_noisy_data, E_init)

def test_PoissonEnergy(field):
    if not isinstance(field, ift.Field):
        return
    if np.iscomplexobj(field.val):
        return
    def get_noisy_data(mean):
        return ift.makeField(mean.domain, np.random.poisson(mean.val))
    lam = 5*field**2 # make rate positive
    E_init = lambda mean : ift.PoissonianEnergy(mean)
    energy_tester(lam, get_noisy_data, E_init)

