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
from nifty8 import myassert
from numpy.testing import assert_allclose, assert_raises

from .common import setup_function, teardown_function

pmp = pytest.mark.parametrize


@pmp('constants', ([], ['a'], ['b'], ['a', 'b']))
@pmp('point_estimates', ([], ['a'], ['b'], ['a', 'b']))
@pmp('mirror_samples', (True, False))
@pmp('mf', (True, False))
@pmp('geo', (True, False))
def test_kl(constants, point_estimates, mirror_samples, mf, geo):
    if not mf and (len(point_estimates) != 0 or len(constants) != 0):
        return
    dom = ift.RGSpace((12,), (2.12))
    op = ift.HarmonicSmoothingOperator(dom, 3)
    if mf:
        op = ift.ducktape(dom, None, 'a')*(op.ducktape('b'))
    lh = ift.GaussianEnergy(domain=op.target, sampling_dtype=np.float64) @ op
    ic = ift.GradientNormController(iteration_limit=5)
    ic.enable_logging()
    h = ift.StandardHamiltonian(lh, ic_samp=ic)
    mean0 = ift.from_random(h.domain, 'normal')

    nsamps = 2
    args = {'constants': constants,
            'point_estimates': point_estimates,
            'mirror_samples': mirror_samples,
            'n_samples': nsamps,
            'position': mean0,
            'hamiltonian': h,
            'minimizer_sampling': ift.NewtonCG(ic) if geo else None}
    if isinstance(mean0, ift.MultiField) and set(point_estimates) == set(mean0.keys()):
        with assert_raises(RuntimeError):
            ift.SampledKLEnergy(**args)
        return
    kl = ift.SampledKLEnergy(**args)
    myassert(len(ic.history) > 0)
    myassert(len(ic.history) == len(ic.history.time_stamps))
    myassert(len(ic.history) == len(ic.history.energy_values))
    ic.history.reset()
    myassert(len(ic.history) == 0)
    myassert(len(ic.history) == len(ic.history.time_stamps))
    myassert(len(ic.history) == len(ic.history.energy_values))

    samp = kl._sample_list
    if isinstance(mean0, ift.MultiField):
        invariant = list(set(constants).intersection(point_estimates))
        _, tmph = h.simplify_for_constant_input(mean0.extract_by_keys(invariant))
        tmpmean = mean0.extract(tmph.domain)
        invariant = mean0.extract_by_keys(invariant)
    else:
        tmph = h
        tmpmean = mean0
        invariant = None
    ift.extra.assert_equal(samp._m, tmpmean)

    from nifty8.minimization.kl_energies import SampledKLEnergyClass
    klpure = SampledKLEnergyClass(samp, tmph, constants, invariant, False)
    # Test number of samples
    expected_nsamps = 2*nsamps if mirror_samples else nsamps
    myassert(kl.samples.n_samples == expected_nsamps)

    # Test value
    assert_allclose(kl.value, klpure.value)

    # Test gradient
    if not mf:
        ift.extra.assert_allclose(kl.gradient, klpure.gradient, 0, 1e-14)
        return

    for kk in kl.position.domain.keys():
        res1 = kl.gradient[kk].val
        if kk in constants:
            res0 = 0*res1
        else:
            res0 = klpure.gradient[kk].val
        assert_allclose(res0, res1)


@pmp('mirror_samples', (True, False))
@pmp('fc', (True, False))
def test_ParametricVI(mirror_samples, fc):
    dom = ift.RGSpace((12,), (2.12))
    op = ift.HarmonicSmoothingOperator(dom, 3)
    op = ift.ducktape(dom, None, 'a')*(op.ducktape('b'))
    lh = ift.GaussianEnergy(domain=op.target, sampling_dtype=np.float64) @ op
    ic = ift.GradientNormController(iteration_limit=5)
    ic.enable_logging()
    h = ift.StandardHamiltonian(lh, ic_samp=ic)
    initial_mean = ift.from_random(h.domain, 'normal')
    nsamps = 10
    args = initial_mean, h, nsamps, mirror_samples, 0.01
    model = (ift.FullCovarianceVI if fc else ift.MeanFieldVI)(*args)
    kl = model.KL
    expected_nsamps = 2*nsamps if mirror_samples else nsamps
    myassert(len(tuple(kl._local_ops)) == expected_nsamps)

    true_val = []
    for i in range(expected_nsamps):
        lat_rnd = ift.from_random(model.KL._op.domain['latent'])
        samp = kl.position.to_dict()
        samp['latent'] = lat_rnd
        samp = ift.MultiField.from_dict(samp)
        true_val.append(model.KL._op(samp))
    true_val = sum(true_val)/expected_nsamps
    assert_allclose(true_val.val, kl.value, rtol=0.1)

    samples = model.KL.samples()
    ift.random.current_rng().integers(0, 100)
    samples1 = model.KL.samples()
    for aa, bb in zip(samples, samples1):
        ift.extra.assert_allclose(aa, bb)
