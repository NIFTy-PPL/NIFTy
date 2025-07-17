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

import nifty.cl as ift
import numpy as np
import pytest
from mpi4py import MPI
from numpy.testing import assert_equal, assert_raises

from ..common import setup_function, teardown_function

comm = MPI.COMM_WORLD
ntask = comm.Get_size()
rank = comm.Get_rank()
master = (rank == 0)
mpi = ntask > 1

if not mpi:
    comm = None

pmp = pytest.mark.parametrize
pms = pytest.mark.skipif


@pms(ntask != 2, reason="requires exactly two mpi tasks")
@pmp('constants', ([], ['a'], ['b'], ['a', 'b']))
@pmp('point_estimates', ([], ['a'], ['b'], ['a', 'b']))
@pmp('mirror_samples', (False, True))
@pmp('mf', (False, True))
@pmp('geo', (False, True))
@pmp('nsamps', (1, 2))
def test_kl(constants, point_estimates, mirror_samples, mf, geo, nsamps):
    if not mf and (len(point_estimates) != 0 or len(constants) != 0):
        return
    dom = ift.RGSpace((12,), (2.12))
    op = ift.HarmonicSmoothingOperator(dom, 3)
    if mf:
        op = ift.ducktape(dom, None, 'a')*(op.ducktape('b'))
    lh = ift.GaussianEnergy(domain=op.target, sampling_dtype=np.float64) @ op
    ic = ift.GradientNormController(iteration_limit=5)
    ic2 = ift.GradientNormController(iteration_limit=5)
    h = ift.StandardHamiltonian(lh, ic_samp=ic, prior_sampling_dtype=float)
    mean0 = ift.from_random(h.domain, 'normal')
    args = {'constants': constants,
            'point_estimates': point_estimates,
            'mirror_samples': mirror_samples,
            'n_samples': nsamps,
            'position': mean0,
            'hamiltonian': h,
            'minimizer_sampling': ift.NewtonCG(ic2) if geo else None}
    if isinstance(mean0, ift.MultiField) and set(point_estimates) == set(mean0.keys()):
        with assert_raises(RuntimeError):
            ift.SampledKLEnergy(**args, comm=comm)
        return

    kl0 = ift.SampledKLEnergy(**args, comm=comm)
    if isinstance(mean0, ift.MultiField):
        invariant = list(set(constants).intersection(point_estimates))
        _, tmph = h.simplify_for_constant_input(mean0.extract_by_keys(invariant))
        tmpmean = mean0.extract(tmph.domain)
        invariant = mean0.extract_by_keys(invariant)
    else:
        tmph = h
        tmpmean = mean0
        invariant = None
    samp = kl0._sample_list
    ift.extra.assert_allclose(tmpmean, samp._m)

    if not mpi:
        samples = tuple(s for s in samp._r)
        ii = len(samples)//2
        slc = slice(None, ii) if rank == 0 else slice(ii, None)
        locsamp = samples[slc]
        if mirror_samples:
            neg = [False, ]*2*nsamps if geo else [False, True]*nsamps
        else:
            neg = [False, ]*nsamps
        locneg = neg[slc]
        samp = ift.minimization.sample_list.ResidualSampleList(
                    tmpmean, locsamp, locneg, comm)

    from nifty.cl.minimization.kl_energies import SampledKLEnergyClass
    kl1 = SampledKLEnergyClass(samp, tmph, constants, invariant, False)

    # Test number of samples
    expected_nsamps = 2*nsamps if mirror_samples else nsamps
    ift.myassert(kl0.samples.n_samples == expected_nsamps)
    ift.myassert(kl1.samples.n_samples == expected_nsamps)

    # Test value
    assert_equal(kl0.value, kl1.value)

    # Test gradient
    if mf:
        for kk in kl0.gradient.domain.keys():
            res0 = kl0.gradient[kk].asnumpy()
            res1 = kl1.gradient[kk].asnumpy()
            assert_equal(res0, res1)
    else:
        assert_equal(kl0.gradient.asnumpy(), kl1.gradient.asnumpy())


@pmp('seed', (42, 123))
@pmp('n_samples', (1, 2, 5, 6))
@pmp('geo', (False, True))
def test_mirror(n_samples, seed, geo):
    ift.random.push_sseq_from_seed(seed)
    a = ift.FieldAdapter(ift.UnstructuredDomain(2), 'a').exp()
    lh = ift.GaussianEnergy(domain=a.target, sampling_dtype=float) @ a
    H = ift.StandardHamiltonian(lh, ic_samp=ift.AbsDeltaEnergyController(1E-10, iteration_limit=2),
                                prior_sampling_dtype=float)
    mini = None
    if geo:
        mini = ift.NewtonCG(ift.AbsDeltaEnergyController(1E-10, iteration_limit=0))
    KL = ift.SampledKLEnergy(ift.from_random(H.domain), H, n_samples, mini,
                             mirror_samples=True, comm=comm)
    sams = list([s-KL.position for s in KL.samples.iterator()])
    for i in range(len(sams)//2):
        ift.extra.assert_allclose(sams[2*i], -sams[2*i+1])


def test_complex_sampling():
    position_space = ift.RGSpace([10, 10])
    args = {
        'offset_mean': 0,
        'offset_std': (1e-3, 1e-6),
        'fluctuations': (1., 0.8),
        'loglogavgslope': (-3., 1),
        'flexibility': (2, 1.),
        'asperity': (0.5, 0.4),
    }
    correlated_field = ift.SimpleCorrelatedField(position_space, **args)
    signal = ift.sigmoid(correlated_field)
    n_los = 100
    LOS_starts = list(ift.random.current_rng().random((n_los, 2)).T)
    LOS_ends = list(ift.random.current_rng().random((n_los, 2)).T)
    R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
    signal_response = R(signal)
    data_space = R.target
    noise = .001
    N = ift.ScalingOperator(data_space, noise, np.float64)
    mock_position = ift.from_random(signal_response.domain, 'normal')
    data = signal_response(mock_position) + N.draw_sample()
    ic_sampling = ift.AbsDeltaEnergyController(deltaE=0.05, iteration_limit=0)
    likelihood_energy = (ift.GaussianEnergy(data, inverse_covariance=N.inverse) @
                         signal_response)
    dtype = {kk: float for kk in likelihood_energy.domain.keys()}
    dtype["xi"] = complex
    pos = ift.from_random(likelihood_energy.domain, dtype=dtype)
    ham = ift.StandardHamiltonian(likelihood_energy, ic_sampling, dtype)
    e = ift.SampledKLEnergy(pos, ham, 100, None)

    ref, tol = 1., 1e-1
    val = np.std(list(e.samples.iterator(lambda x: x["xi"].val.real[0, 0].asnumpy())))
    assert abs(ref-val)/(abs(ref)+abs(val)) < tol
    val = np.std(list(e.samples.iterator(lambda x: x["xi"].val.imag[0, 0].asnumpy())))
    assert abs(ref-val)/(abs(ref)+abs(val)) < tol
