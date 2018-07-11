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

from __future__ import division

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


class HarmonicTransformOperatorTests(unittest.TestCase):
    @expand(product([128, 256],
                    [np.float64, np.float32, np.complex64, np.complex128]))
    def test_dotsht(self, lm, tp):
        tol = 10 * _get_rtol(tp)
        a = ift.LMSpace(lmax=lm)
        b = ift.GLSpace(nlat=lm+1)
        fft = ift.HarmonicTransformOperator(domain=a, target=b)
        inp = ift.Field.from_random(domain=a, random_type='normal',
                                    std=1, mean=0, dtype=tp)
        out = fft.times(inp)
        v1 = np.sqrt(out.vdot(out))
        v2 = np.sqrt(inp.vdot(fft.adjoint_times(out)))
        assert_allclose(v1, v2, rtol=tol, atol=tol)

    @expand(product([128, 256],
                    [np.float64, np.float32, np.complex64, np.complex128]))
    def test_dotsht2(self, lm, tp):
        tol = 10 * _get_rtol(tp)
        a = ift.LMSpace(lmax=lm)
        b = ift.HPSpace(nside=lm//2)
        fft = ift.HarmonicTransformOperator(domain=a, target=b)
        inp = ift.Field.from_random(domain=a, random_type='normal',
                                    std=1, mean=0, dtype=tp)
        out = fft.times(inp)
        v1 = np.sqrt(out.vdot(out))
        v2 = np.sqrt(inp.vdot(fft.adjoint_times(out)))
        assert_allclose(v1, v2, rtol=tol, atol=tol)

    @expand(product([ift.LMSpace(lmax=30, mmax=25)],
                    [np.float64, np.float32, np.complex64, np.complex128]))
    def test_normalisation(self, space, tp):
        tol = 10 * _get_rtol(tp)
        cospace = space.get_default_codomain()
        fft = ift.HarmonicTransformOperator(space, cospace)
        inp = ift.Field.from_random(domain=space, random_type='normal',
                                    std=1, mean=2, dtype=tp)
        out = fft.times(inp)
        zero_idx = tuple([0]*len(space.shape))
        assert_allclose(inp.to_global_data()[zero_idx], out.integrate(),
                        rtol=tol, atol=tol)
