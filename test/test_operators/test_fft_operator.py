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
import numpy as np
from numpy.testing import assert_allclose
import nifty2go as ift
from itertools import product
from test.common import expand


def _get_rtol(tp):
    if (tp == np.float64) or (tp == np.complex128):
        return 1e-10
    else:
        return 1e-5


class FFTOperatorTests(unittest.TestCase):
    @expand(product([16, ], [0.1, 1, 3.7],
                    [np.float64, np.float32, np.complex64, np.complex128]))
    def test_fft1D(self, dim1, d, itp):
        tol = _get_rtol(itp)
        a = ift.RGSpace(dim1, distances=d)
        b = ift.RGSpace(dim1, distances=1./(dim1*d), harmonic=True)
        fft = ift.FFTOperator(domain=a, target=b)

        np.random.seed(16)
        inp = ift.Field.from_random(domain=a, random_type='normal',
                                    std=7, mean=3, dtype=itp)
        out = fft.adjoint_times(fft.times(inp))
        assert_allclose(ift.dobj.to_global_data(inp.val),
                        ift.dobj.to_global_data(out.val), rtol=tol, atol=tol)

    @expand(product([12, 15], [9, 12], [0.1, 1, 3.7],
                    [0.4, 1, 2.7],
                    [np.float64, np.float32, np.complex64, np.complex128]))
    def test_fft2D(self, dim1, dim2, d1, d2, itp):
        tol = _get_rtol(itp)
        a = ift.RGSpace([dim1, dim2], distances=[d1, d2])
        b = ift.RGSpace([dim1, dim2],
                        distances=[1./(dim1*d1), 1./(dim2*d2)], harmonic=True)
        fft = ift.FFTOperator(domain=a, target=b)

        inp = ift.Field.from_random(domain=a, random_type='normal',
                                    std=7, mean=3, dtype=itp)
        out = fft.adjoint_times(fft.times(inp))
        assert_allclose(ift.dobj.to_global_data(inp.val),
                        ift.dobj.to_global_data(out.val), rtol=tol, atol=tol)

    @expand(product([0, 1, 2],
                    [np.float64, np.float32, np.complex64, np.complex128]))
    def test_composed_fft(self, index, dtype):
        tol = _get_rtol(dtype)
        a = [a1, a2, a3] = [ift.RGSpace((32,)), ift.RGSpace((4, 4)),
                            ift.RGSpace((5, 6))]
        fft = ift.FFTOperator(domain=a, space=index)

        inp = ift.Field.from_random(domain=(a1, a2, a3), random_type='normal',
                                    std=7, mean=3, dtype=dtype)
        out = fft.adjoint_times(fft.times(inp))
        assert_allclose(ift.dobj.to_global_data(inp.val),
                        ift.dobj.to_global_data(out.val), rtol=tol, atol=tol)

    @expand(product([0, 3, 6, 11, 30],
                    [np.float64, np.float32, np.complex64, np.complex128]))
    def test_sht(self, lm, tp):
        tol = _get_rtol(tp)
        a = ift.LMSpace(lmax=lm)
        b = ift.GLSpace(nlat=lm+1)
        fft = ift.FFTOperator(domain=a, target=b)
        inp = ift.Field.from_random(domain=a, random_type='normal',
                                    std=7, mean=3, dtype=tp)
        out = fft.adjoint_times(fft.times(inp))
        assert_allclose(ift.dobj.to_global_data(inp.val),
                        ift.dobj.to_global_data(out.val), rtol=tol, atol=tol)

    @expand(product([128, 256],
                    [np.float64, np.float32, np.complex64, np.complex128]))
    def test_sht2(self, lm, tp):
        a = ift.LMSpace(lmax=lm)
        b = ift.HPSpace(nside=lm//2)
        fft = ift.FFTOperator(domain=a, target=b)
        inp = ift.Field.from_random(domain=a, random_type='normal',
                                    std=1, mean=0, dtype=tp)
        out = fft.adjoint_times(fft.times(inp))
        assert_allclose(ift.dobj.to_global_data(inp.val),
                        ift.dobj.to_global_data(out.val), rtol=1e-3, atol=1e-1)

    @expand(product([128, 256],
                    [np.float64, np.float32, np.complex64, np.complex128]))
    def test_dotsht(self, lm, tp):
        tol = 10 * _get_rtol(tp)
        a = ift.LMSpace(lmax=lm)
        b = ift.GLSpace(nlat=lm+1)
        fft = ift.FFTOperator(domain=a, target=b)
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
        fft = ift.FFTOperator(domain=a, target=b)
        inp = ift.Field.from_random(domain=a, random_type='normal',
                                    std=1, mean=0, dtype=tp)
        out = fft.times(inp)
        v1 = np.sqrt(out.vdot(out))
        v2 = np.sqrt(inp.vdot(fft.adjoint_times(out)))
        assert_allclose(v1, v2, rtol=tol, atol=tol)
