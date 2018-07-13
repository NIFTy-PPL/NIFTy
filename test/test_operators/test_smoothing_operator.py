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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import unittest
from itertools import product
from test.common import expand

import nifty5 as ift
import numpy as np
from numpy.testing import assert_allclose


def _get_rtol(tp):
    if (tp == np.float64) or (tp == np.complex128):
        return 1e-10
    else:
        return 1e-5


class SmoothingOperator_Tests(unittest.TestCase):
    spaces = [ift.RGSpace(128)]

    @expand(product(spaces, [0., .5, 5.]))
    def test_property(self, space, sigma):
        op = ift.HarmonicSmoothingOperator(space, sigma=sigma)
        if op.domain[0] != space:
            raise TypeError

    @expand(product(spaces, [0., .5, 5.]))
    def test_adjoint_times(self, space, sigma):
        op = ift.HarmonicSmoothingOperator(space, sigma=sigma)
        rand1 = ift.Field.from_random('normal', domain=space)
        rand2 = ift.Field.from_random('normal', domain=space)
        tt1 = rand1.vdot(op.times(rand2))
        tt2 = rand2.vdot(op.adjoint_times(rand1))
        assert_allclose(tt1, tt2)

    @expand(product(spaces, [0., .5, 5.]))
    def test_times(self, space, sigma):
        op = ift.HarmonicSmoothingOperator(space, sigma=sigma)
        fld = np.zeros(space.shape, dtype=np.float64)
        fld[0] = 1.
        rand1 = ift.Field.from_global_data(space, fld)
        tt1 = op.times(rand1)
        assert_allclose(1, tt1.sum())

    @expand(product([128, 256], [1, 0.4], [0., 1.,  3.7],
                    [np.float64, np.complex128]))
    def test_smooth_regular1(self, sz, d, sigma, tp):
        tol = _get_rtol(tp)
        sp = ift.RGSpace(sz, distances=d)
        smo = ift.HarmonicSmoothingOperator(sp, sigma=sigma)
        inp = ift.Field.from_random(domain=sp, random_type='normal', std=1,
                                    mean=4, dtype=tp)
        out = smo(inp)
        assert_allclose(inp.sum(), out.sum(), rtol=tol, atol=tol)

    @expand(product([10, 15], [7, 10], [1, 0.4], [2, 0.3], [0., 1.,  3.7],
                    [np.float64, np.complex128]))
    def test_smooth_regular2(self, sz1, sz2, d1, d2, sigma, tp):
        tol = _get_rtol(tp)
        sp = ift.RGSpace([sz1, sz2], distances=[d1, d2])
        smo = ift.HarmonicSmoothingOperator(sp, sigma=sigma)
        inp = ift.Field.from_random(domain=sp, random_type='normal', std=1,
                                    mean=4, dtype=tp)
        out = smo(inp)
        assert_allclose(inp.sum(), out.sum(), rtol=tol, atol=tol)
