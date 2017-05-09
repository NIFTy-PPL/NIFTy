# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
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

import unittest

import numpy as np
from numpy.testing import assert_equal,\
    assert_allclose
from nifty.config import dependency_injector as di

from nifty import Field,\
    RGSpace,\
    LMSpace,\
    HPSpace,\
    GLSpace,\
    FFTOperator

from itertools import product
from test.common import expand

from nose.plugins.skip import SkipTest


def _harmonic_type(itp):
    otp = itp
    if otp == np.float64:
        otp = np.complex128
    elif otp == np.float32:
        otp = np.complex64
    return otp


def _get_rtol(tp):
    if (tp == np.float64) or (tp == np.complex128):
        return 1e-10
    else:
        return 1e-5


class Misc_Tests(unittest.TestCase):
    @expand(product([10, 11], [False, True], [0.1, 1, 3.7]))
    def test_RG_distance_1D(self, dim1, zc1, d):
        foo = RGSpace([dim1], zerocenter=zc1, distances=d)
        res = foo.get_distance_array('not')
        assert_equal(res[zc1 * (dim1 // 2)], 0.)

    @expand(product([10, 11], [9, 28], [False, True], [False, True],
                    [0.1, 1, 3.7]))
    def test_RG_distance_2D(self, dim1, dim2, zc1, zc2, d):
        foo = RGSpace([dim1, dim2], zerocenter=[zc1, zc2], distances=d)
        res = foo.get_distance_array('not')
        assert_equal(res[zc1 * (dim1 // 2), zc2 * (dim2 // 2)], 0.)

    @expand(product(["numpy", "fftw"], [10, 11], [False, True], [False, True],
                    [0.1, 1, 3.7],
                    [np.float64, np.complex128, np.float32, np.complex64]))
    def test_fft1D(self, module, dim1, zc1, zc2, d, itp):
        if module == "fftw" and "pyfftw" not in di:
            raise SkipTest
        tol = _get_rtol(itp)
        a = RGSpace(dim1, zerocenter=zc1, distances=d)
        b = RGSpace(dim1, zerocenter=zc2,distances=1./(dim1*d),harmonic=True)
        fft = FFTOperator(domain=a, target=b, domain_dtype=itp,
                          target_dtype=_harmonic_type(itp), module=module)
        inp = Field.from_random(domain=a, random_type='normal', std=7, mean=3,
                                dtype=itp)
        out = fft.adjoint_times(fft.times(inp))
        assert_allclose(inp.val, out.val, rtol=tol, atol=tol)

    @expand(product(["numpy", "fftw"], [10, 11], [9, 12], [False, True],
                    [False, True], [False, True], [False, True], [0.1, 1, 3.7],
                    [0.4, 1, 2.7],
                    [np.float64, np.complex128, np.float32, np.complex64]))
    def test_fft2D(self, module, dim1, dim2, zc1, zc2, zc3, zc4, d1, d2, itp):
        if module == "fftw" and "pyfftw" not in di:
            raise SkipTest
        tol = _get_rtol(itp)
        a = RGSpace([dim1, dim2], zerocenter=[zc1, zc2], distances=[d1,d2])
        b = RGSpace([dim1, dim2], zerocenter=[zc3, zc4],
                    distances=[1./(dim1*d1),1./(dim2*d2)],harmonic=True)
        fft = FFTOperator(domain=a, target=b, domain_dtype=itp,
                          target_dtype=_harmonic_type(itp), module=module)
        inp = Field.from_random(domain=a, random_type='normal', std=7, mean=3,
                                dtype=itp)
        out = fft.adjoint_times(fft.times(inp))
        assert_allclose(inp.val, out.val, rtol=tol, atol=tol)

    @expand(product([0, 3, 6, 11, 30],
                    [np.float64, np.complex128, np.float32, np.complex64]))
    def test_sht(self, lm, tp):
        if 'pyHealpix' not in di:
            raise SkipTest
        tol = _get_rtol(tp)
        a = LMSpace(lmax=lm)
        b = GLSpace(nlat=lm+1)
        fft = FFTOperator(domain=a, target=b, domain_dtype=tp, target_dtype=tp)
        inp = Field.from_random(domain=a, random_type='normal', std=7, mean=3,
                                dtype=tp)
        out = fft.adjoint_times(fft.times(inp))
        assert_allclose(inp.val, out.val, rtol=tol, atol=tol)

    @expand(product([128, 256],
                    [np.float64, np.complex128, np.float32, np.complex64]))
    def test_sht2(self, lm, tp):
        if 'pyHealpix' not in di:
            raise SkipTest
        a = LMSpace(lmax=lm)
        b = HPSpace(nside=lm//2)
        fft = FFTOperator(domain=a, target=b, domain_dtype=tp, target_dtype=tp)
        inp = Field.from_random(domain=a, random_type='normal', std=1, mean=0,
                                dtype=tp)
        out = fft.adjoint_times(fft.times(inp))
        assert_allclose(inp.val, out.val, rtol=1e-3, atol=1e-1)

    @expand(product([128, 256],
                    [np.float64, np.complex128, np.float32, np.complex64]))
    def test_dotsht(self, lm, tp):
        if 'pyHealpix' not in di:
            raise SkipTest
        tol = _get_rtol(tp)
        a = LMSpace(lmax=lm)
        b = GLSpace(nlat=lm+1)
        fft = FFTOperator(domain=a, target=b, domain_dtype=tp, target_dtype=tp)
        inp = Field.from_random(domain=a, random_type='normal', std=1, mean=0,
                                dtype=tp)
        out = fft.times(inp)
        v1=np.sqrt(out.dot(out))
        v2=np.sqrt(inp.dot(fft.adjoint_times(out)))
        assert_allclose(v1,v2, rtol=tol, atol=tol)

    @expand(product([128, 256],
                    [np.float64, np.complex128, np.float32, np.complex64]))
    def test_dotsht2(self, lm, tp):
        if 'pyHealpix' not in di:
            raise SkipTest
        tol = _get_rtol(tp)
        a = LMSpace(lmax=lm)
        b = HPSpace(nside=lm//2)
        fft = FFTOperator(domain=a, target=b, domain_dtype=tp, target_dtype=tp)
        inp = Field.from_random(domain=a, random_type='normal', std=1, mean=0,
                                dtype=tp)
        out = fft.times(inp)
        v1=np.sqrt(out.dot(out))
        v2=np.sqrt(inp.dot(fft.adjoint_times(out)))
        assert_allclose(v1,v2, rtol=tol, atol=tol)
