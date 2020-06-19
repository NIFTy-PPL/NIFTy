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

import nifty7 as ift

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


def _complex2real(sp):
    tup = tuple([d for d in sp])
    rsp = ift.DomainTuple.make((ift.UnstructuredDomain(2),) + tup)
    rl = ift.DomainTupleFieldInserter(rsp, 0, (0,))
    im = ift.DomainTupleFieldInserter(rsp, 0, (1,))
    x = ift.ScalingOperator(sp, 1)
    return rl(x.real)+im(x.imag)


def test_complex2real():
    sp = ift.UnstructuredDomain(3)
    op = _complex2real(ift.makeDomain(sp))
    f = ift.from_random(op.domain, 'normal', dtype=np.complex128)
    assert np.all((f == op.adjoint_times(op(f))).val)
    assert op(f).dtype == np.float64
    f = ift.from_random(op.target, 'normal')
    assert np.all((f == op(op.adjoint_times(f))).val)


def energy_tester(pos, get_noisy_data, energy_initializer):
    if isinstance(pos, ift.Field):
        if np.issubdtype(pos.dtype, np.complexfloating):
            op = _complex2real(pos.domain)
        else:
            op = ift.ScalingOperator(pos.domain, 1.)
    else:
        ops = []
        for k,dom in pos.domain.items():
            if np.issubdtype(pos[k].dtype, np.complexfloating):
                ops.append(_complex2real(dom).ducktape(k).ducktape_left(k))
            else:
                FA = ift.FieldAdapter(dom, k)
                ops.append(FA.adjoint @ FA)
        realizer = ift.utilities.my_sum(ops)
        from nifty7.operator_spectrum import _DomRemover
        flattener = _DomRemover(realizer.target)
        op = flattener @ realizer

    npos = op(pos)
    nget_noisy_data = lambda mean: get_noisy_data(op.adjoint_times(mean))
    nenergy_initializer = lambda mean: energy_initializer(mean) @ op.adjoint
    _actual_energy_tester(npos, nget_noisy_data, nenergy_initializer)


def _actual_energy_tester(pos, get_noisy_data, energy_initializer):
    domain = pos.domain
    test_vec = ift.from_random(domain, 'normal')
    results = []
    lin = ift.Linearization.make_var(pos)
    for i in range(Nsamp):
        data = get_noisy_data(pos)
        energy = energy_initializer(data)
        grad = energy(lin).jac.adjoint(ift.full(energy.target, 1.))
        results.append(_to_array((grad*grad.s_vdot(test_vec)).val))
    print(energy)
    print(grad)
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
    get_noisy_data = lambda mean: mean + icov.draw_sample_with_dtype(
        from_inverse=True, dtype=dtype)
    E_init = lambda mean: ift.GaussianEnergy(mean=mean, inverse_covariance=icov)
    energy_tester(field, get_noisy_data, E_init)


def test_PoissonEnergy(field):
    if not isinstance(field, ift.Field):
        pytest.skip("MultiField Poisson energy  not supported")
    if np.iscomplexobj(field.val):
        pytest.skip("Poisson energy not defined for complex flux")
    get_noisy_data = lambda mean: ift.makeField(mean.domain, np.random.poisson(mean.val))
    # Make rate positive and high enough to avoid bad statistic
    lam = 10*(field**2).clip(0.1, None)
    E_init = lambda mean: ift.PoissonianEnergy(mean)
    energy_tester(lam, get_noisy_data, E_init)
