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

import nifty6 as ift
from numpy.testing import assert_, assert_allclose
import pytest
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
    h = ift.StandardHamiltonian(lh, ic_samp=ic)
    mean0 = ift.from_random('normal', h.domain)

    nsamps = 2
    kl = ift.MetricGaussianKL(mean0,
                              h,
                              nsamps,
                              constants=constants,
                              point_estimates=point_estimates,
                              mirror_samples=mirror_samples,
                              napprox=0)
    locsamp = kl._local_samples
    klpure = ift.MetricGaussianKL(mean0,
                                  h,
                                  nsamps,
                                  mirror_samples=mirror_samples,
                                  napprox=0,
                                  _local_samples=locsamp)

    # Test number of samples
    expected_nsamps = 2*nsamps if mirror_samples else nsamps
    assert_(len(tuple(kl.samples)) == expected_nsamps)

    # Test value
    assert_allclose(kl.value, klpure.value)

    # Test gradient
    if not mf:
        ift.extra.assert_allclose(kl.gradient, klpure.gradient, 0, 1e-14)
        return

    for kk in h.domain.keys():
        res0 = klpure.gradient[kk].val
        if kk in constants:
            res0 = 0*res0
        res1 = kl.gradient[kk].val
        assert_allclose(res0, res1)

    # Test point_estimates (after drawing samples)
    for kk in point_estimates:
        for ss in kl.samples:
            ss = ss[kk].val
            assert_allclose(ss, 0*ss)

    # Test constants (after some minimization)
    cg = ift.GradientNormController(iteration_limit=5)
    minimizer = ift.NewtonCG(cg)
    kl, _ = minimizer(kl)
    diff = (mean0 - kl.position).to_dict()
    for kk in constants:
        assert_allclose(diff[kk].val, 0*diff[kk].val)
