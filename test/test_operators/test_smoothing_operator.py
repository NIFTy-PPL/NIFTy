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

import nifty8 as ift
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..common import list2fixture, setup_function, teardown_function


def _get_rtol(tp):
    if (tp == np.float64) or (tp == np.complex128):
        return 1e-10
    else:
        return 1e-5


space = list2fixture([ift.RGSpace(128)])
sigma = list2fixture([0., .5, 5.])
tp = list2fixture([np.float64, np.complex128])
pmp = pytest.mark.parametrize


def test_property(space, sigma):
    op = ift.HarmonicSmoothingOperator(space, sigma=sigma)
    if op.domain[0] != space:
        raise TypeError


def test_adjoint_times(space, sigma):
    op = ift.HarmonicSmoothingOperator(space, sigma=sigma)
    rand1 = ift.Field.from_random(domain=space, random_type='normal')
    rand2 = ift.Field.from_random(domain=space, random_type='normal')
    tt1 = rand1.s_vdot(op.times(rand2))
    tt2 = rand2.s_vdot(op.adjoint_times(rand1))
    assert_allclose(tt1, tt2)


def test_times(space, sigma):
    op = ift.HarmonicSmoothingOperator(space, sigma=sigma)
    fld = np.zeros(space.shape, dtype=np.float64)
    fld[0] = 1.
    rand1 = ift.Field.from_raw(space, fld)
    tt1 = op.times(rand1)
    assert_allclose(1, tt1.s_sum())


@pmp('sz', [128, 256])
@pmp('d', [1, 0.4])
def test_smooth_regular1(sz, d, sigma, tp):
    tol = _get_rtol(tp)
    sp = ift.RGSpace(sz, distances=d)
    smo = ift.HarmonicSmoothingOperator(sp, sigma=sigma)
    inp = ift.Field.from_random(domain=sp, random_type='normal', dtype=tp, std=1, mean=4)
    out = smo(inp)
    assert_allclose(inp.s_sum(), out.s_sum(), rtol=tol, atol=tol)


@pmp('sz1', [10, 15])
@pmp('sz2', [7, 10])
@pmp('d1', [1, 0.4])
@pmp('d2', [2, 0.3])
def test_smooth_regular2(sz1, sz2, d1, d2, sigma, tp):
    tol = _get_rtol(tp)
    sp = ift.RGSpace([sz1, sz2], distances=[d1, d2])
    smo = ift.HarmonicSmoothingOperator(sp, sigma=sigma)
    inp = ift.Field.from_random(domain=sp, random_type='normal', dtype=tp, std=1, mean=4)
    out = smo(inp)
    assert_allclose(inp.s_sum(), out.s_sum(), rtol=tol, atol=tol)
