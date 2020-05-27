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
from mpi4py import MPI
from numpy.testing import assert_, assert_equal, assert_raises

import nifty6 as ift

from ..common import setup_function, teardown_function

comm = MPI.COMM_WORLD
ntask = comm.Get_size()
rank = comm.Get_rank()
master = (rank == 0)
mpi = ntask > 1

pmp = pytest.mark.parametrize
pms = pytest.mark.skipif


@pms(ntask != 2, reason="requires exactly two mpi tasks")
@pmp('constants', ([], ['a'], ['b'], ['a', 'b']))
@pmp('point_estimates', ([], ['a'], ['b'], ['a', 'b']))
@pmp('mirror_samples', (False, True))
@pmp('mode', (0, 1))
@pmp('mf', (False, True))
def test_kl(constants, point_estimates, mirror_samples, mode, mf):
    if not mf and (len(point_estimates) != 0 or len(constants) != 0):
        return
    dom = ift.RGSpace((12,), (2.12))
    op = ift.HarmonicSmoothingOperator(dom, 3)
    if mf:
        op = ift.ducktape(dom, None, 'a')*(op.ducktape('b'))
    lh = ift.GaussianEnergy(domain=op.target, sampling_dtype=np.float64) @ op
    ic = ift.GradientNormController(iteration_limit=5)
    h = ift.StandardHamiltonian(lh, ic_samp=ic)
    mean0 = ift.from_random(h.domain, 'normal')
    nsamps = 2
    args = {'constants': constants,
            'point_estimates': point_estimates,
            'mirror_samples': mirror_samples,
            'n_samples': 2,
            'mean': mean0,
            'hamiltonian': h}
    if isinstance(mean0, ift.MultiField) and set(point_estimates) == set(mean0.keys()):
        with assert_raises(RuntimeError):
            ift.MetricGaussianKL(**args, comm=comm)
        return
    if mode == 0:
        kl0 = ift.MetricGaussianKL(**args, comm=comm)
        locsamp = kl0._local_samples
        kl1 = ift.MetricGaussianKL(**args, comm=comm, _local_samples=locsamp)
    elif mode == 1:
        kl0 = ift.MetricGaussianKL(**args)
        samples = kl0._local_samples
        ii = len(samples)//2
        slc = slice(None, ii) if rank == 0 else slice(ii, None)
        locsamp = samples[slc]
        kl1 = ift.MetricGaussianKL(**args, comm=comm, _local_samples=locsamp)

    # Test number of samples
    expected_nsamps = 2*nsamps if mirror_samples else nsamps
    assert_(len(tuple(kl0.samples)) == expected_nsamps)
    assert_(len(tuple(kl1.samples)) == expected_nsamps)

    # Test value
    assert_equal(kl0.value, kl1.value)

    # Test gradient
    if mf:
        for kk in h.domain.keys():
            res0 = kl0.gradient[kk].val
            if kk in constants:
                res0 = 0*res0
            res1 = kl1.gradient[kk].val
            assert_equal(res0, res1)
    else:
        assert_equal(kl0.gradient.val, kl1.gradient.val)

    # Test point_estimates (after drawing samples)
    if mf:
        for kk in point_estimates:
            for ss in kl0.samples:
                ss = ss[kk].val
                assert_equal(ss, 0*ss)
            for ss in kl1.samples:
                ss = ss[kk].val
                assert_equal(ss, 0*ss)

    # Test constants (after some minimization)
    if mf:
        cg = ift.GradientNormController(iteration_limit=5)
        minimizer = ift.NewtonCG(cg)
        for e in [kl0, kl1]:
            e, _ = minimizer(e)
            diff = (mean0 - e.position).to_dict()
            for kk in constants:
                assert_equal(diff[kk].val, 0*diff[kk].val)
