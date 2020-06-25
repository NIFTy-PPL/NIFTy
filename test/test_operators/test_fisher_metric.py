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

dtype = list2fixture([np.float64,
                     np.complex128])
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
        for k, dom in pos.domain.items():
            if np.issubdtype(pos[k].dtype, np.complexfloating):
                ops.append(_complex2real(dom).ducktape(k).ducktape_left(k))
            else:
                FA = ift.FieldAdapter(dom, k)
                ops.append(FA.adjoint @ FA)
        realizer = ift.utilities.my_sum(ops)
        from nifty7.operator_spectrum import _DomRemover
        flattener = _DomRemover(realizer.target)
        op = flattener @ realizer
    pos = op(pos)

    domain = pos.domain
    test_vec = ift.from_random(domain, 'normal')
    results = []
    lin = ift.Linearization.make_var(pos)
    for i in range(Nsamp):
        data = get_noisy_data(op.adjoint_times(pos))
        energy = energy_initializer(data) @ op.adjoint
        grad = energy(lin).gradient
        results.append(_to_array((grad*grad.s_vdot(test_vec)).val))
    res = np.mean(np.array(results), axis=0)
    std = np.std(np.array(results), axis=0)/np.sqrt(Nsamp)
    energy = energy_initializer(data) @ op.adjoint
    lin = ift.Linearization.make_var(pos, want_metric=True)
    res2 = _to_array(energy(lin).metric(test_vec).val)
    np.testing.assert_allclose(res/std, res2/std, atol=6)
    for factor in [0.01, 0.5, 2, 100]:
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(res/std, factor*res2/std, atol=6)


def test_GaussianEnergy(field):
    dtype = field.dtype
    icov = ift.from_random(field.domain, 'normal')**2
    icov = ift.makeOp(icov)
    get_noisy_data = lambda mean: mean + icov.draw_sample_with_dtype(
        from_inverse=True, dtype=dtype)
    E_init = lambda data: ift.GaussianEnergy(mean=data, inverse_covariance=icov)
    energy_tester(field, get_noisy_data, E_init)


def test_PoissonEnergy(field):
    if not isinstance(field, ift.Field):
        pytest.skip("MultiField Poisson energy  not supported")
    if np.iscomplexobj(field.val):
        pytest.skip("Poisson energy not defined for complex flux")
    get_noisy_data = lambda mean: ift.makeField(mean.domain, np.random.poisson(mean.val))
    # Make rate positive and high enough to avoid bad statistic
    lam = 10*(field**2).clip(0.1, None)
    E_init = lambda data: ift.PoissonianEnergy(data)
    energy_tester(lam, get_noisy_data, E_init)


@pytest.mark.parametrize('factor', [0.01, 0.5, 1, 2, 100])
def test_VariableCovarianceGaussianEnergy(dtype, factor):
    dom = ift.UnstructuredDomain(3)
    res = ift.from_random(dom, 'normal', dtype=dtype)
    ivar = ift.from_random(dom, 'normal')**2+4.
    mf = ift.MultiField.from_dict({'res':res, 'ivar':ivar})
    energy = ift.VariableCovarianceGaussianEnergy(dom, 'res', 'ivar', dtype, _debugging_factor=factor)
    def get_noisy_data(mean):
        samp = ift.from_random(dom, 'normal', dtype)
        samp = samp/mean['ivar'].sqrt()
        return samp + mean['res']
    def E_init(data):
        adder = ift.Adder(ift.MultiField.from_dict({'res':data}), neg=True)
        return energy.partial_insert(adder)
    if factor != 1.:
        with pytest.raises(AssertionError):
            energy_tester(mf, get_noisy_data, E_init)
    else:
        energy_tester(mf, get_noisy_data, E_init)


def normal(dtype, shape):
    return ift.random.Random.normal(dtype, shape)


def test_variablecovenergy_fast(dtype):
    dtype = np.float64  # FIXME
    npix = 2
    shp = (npix,)
    ntries = 200000

    dom = ift.UnstructuredDomain(shp)
    e = ift.VariableCovarianceGaussianEnergy(dom, 'resi', 'icov', dtype)

    # Positions
    resi = normal(dtype, shp)
    icov = normal(np.float64, shp)**2 + 4.
    sig = np.sqrt(1/icov)
    pos = ift.makeField(e.domain, {'resi': resi, 'icov': icov})

    sampled_data = normal(dtype, (ntries, npix))*sig+resi
    grad0 = (pos['resi'].val-sampled_data)*icov
    fac = 0.5
    if np.issubdtype(dtype, np.complexfloating):
        fac = 1
    grad1 = 0.5*np.abs(pos['resi'].val-sampled_data)**2 - fac/pos['icov'].val

    # Test correctness of gradient
    etest = e.partial_insert(ift.Adder(ift.makeField(dom, -sampled_data[0])).ducktape('resi').ducktape_left('resi'))
    eresult = etest(ift.Linearization.make_var(pos))
    grad0ref = eresult.gradient.val['resi']
    grad1ref = eresult.gradient.val['icov']
    np.testing.assert_allclose(grad0[0], grad0ref)
    np.testing.assert_allclose(grad1[0], grad1ref)

    # Apply test vector
    test_vec_resi = normal(dtype, shp)
    test_vec_icov = normal(np.float64, shp)
    metric00 = np.sum(grad0*test_vec_resi, axis=1)[:, None] * grad0.conjugate()
    metric00 = np.mean(metric00, axis=0)
    metric11 = np.sum(grad1*test_vec_icov, axis=1)[:, None] * grad1.conjugate()
    metric11 = np.mean(grad1, axis=0)

    metric00ref = icov*test_vec_resi
    metric11ref = (1/(icov)**2)*test_vec_icov
    metric00std = np.std(metric00, axis=0)
    metric11std = np.std(metric11, axis=0)
    np.testing.assert_allclose(metric00ref/metric00std, metric00/metric00std, atol=0.1)
    # np.testing.assert_allclose(metric11ref/metric11std, metric11/metric11std, atol=0.1)
    for factor in [0.01, 0.5, 2, 100]:
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(metric00ref/metric00std, factor*metric00/metric00std, atol=0.1)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(metric11ref/metric11std, metric11/metric11std, atol=0.1)
