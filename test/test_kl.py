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
from numpy.testing import assert_, assert_allclose, assert_raises

import nifty7 as ift

from .common import setup_function, teardown_function

pmp = pytest.mark.parametrize


@pmp('constants', ([], ['a'], ['b'], ['a', 'b']))
@pmp('point_estimates', ([], ['a'], ['b'], ['a', 'b']))
@pmp('mirror_samples', (True, False))
@pmp('mf', (True, False))
def test_kl(constants, point_estimates, mirror_samples, mf):
    if not mf and (len(point_estimates) != 0 or len(constants) != 0):
        return
    dom = ift.RGSpace((12,), (2.12))
    op = ift.HarmonicSmoothingOperator(dom, 3)
    if mf:
        op = ift.ducktape(dom, None, 'a')*(op.ducktape('b'))
    import numpy as np
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
            'mean': mean0,
            'hamiltonian': h}
    if isinstance(mean0, ift.MultiField) and set(point_estimates) == set(mean0.keys()):
        with assert_raises(RuntimeError):
            ift.MetricGaussianKL.make(**args)
        return
    kl = ift.MetricGaussianKL.make(**args)
    assert_(len(ic.history) > 0)
    assert_(len(ic.history) == len(ic.history.time_stamps))
    assert_(len(ic.history) == len(ic.history.energy_values))
    ic.history.reset()
    assert_(len(ic.history) == 0)
    assert_(len(ic.history) == len(ic.history.time_stamps))
    assert_(len(ic.history) == len(ic.history.energy_values))

    locsamp = kl._local_samples
    if isinstance(mean0, ift.MultiField):
        _, tmph = h.simplify_for_constant_input(mean0.extract_by_keys(constants))
        tmpmean = mean0.extract(tmph.domain)
    else:
        tmph = h
        tmpmean = mean0
    klpure = ift.MetricGaussianKL(tmpmean, tmph, nsamps, mirror_samples, None, locsamp, False, True)

    # Test number of samples
    expected_nsamps = 2*nsamps if mirror_samples else nsamps
    assert_(len(tuple(kl.samples)) == expected_nsamps)

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
