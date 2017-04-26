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
from numpy.testing import assert_,\
                          assert_equal, \
                          assert_allclose
from nifty.config import dependency_injector as di

from nifty import Field,\
                  RGSpace,\
                  LMSpace,\
                  GLSpace,\
                  FieldArray, \
                  RGRGTransformation, \
                  LMGLTransformation, \
                  LMHPTransformation, \
                  FFTOperator

from nose.plugins.skip import SkipTest

def _harmonic_type(itp):
    otp=itp
    if otp==np.float64:
        otp=np.complex128
    elif otp==np.float32:
        otp=np.complex64
    return otp

def _sub_fft1D(module):
    for dim1 in [10,11]:
        for zc1 in [False,True]:
            for zc2 in [False,True]:
                for d in [0.1,1,3.7]:
                    for itp in [np.float64,np.complex128,np.float32,np.complex64]:
                        a = RGSpace(dim1, zerocenter=zc1, distances=d)
                        b = RGRGTransformation.get_codomain(a, zerocenter=zc2)
                        fft = FFTOperator(domain=a, target=b, domain_dtype=itp, target_dtype=_harmonic_type(itp),module=module)
                        inp = Field.from_random(domain=a,random_type='normal',std=7,mean=3,dtype=itp)
                        out = fft.inverse_times(fft.times(inp))
                        assert_allclose(inp.val, out.val)
def _sub_fft2D(module):
  for dim1 in [10,11]:
    for dim2 in [9,12]:
      for zc1 in [False,True]:
        for zc2 in [False,True]:
          for zc3 in [False,True]:
            for zc4 in [False,True]:
              for d in [0.1,1,3.7]:
                for itp in [np.float64,np.complex128,np.float32,np.complex64]:
                  a = RGSpace([dim1,dim2], zerocenter=[zc1,zc2], distances=d)
                  b = RGRGTransformation.get_codomain(a, zerocenter=[zc3,zc4])
                  fft = FFTOperator(domain=a, target=b, domain_dtype=itp, target_dtype=_harmonic_type(itp),module=module)
                  inp = Field.from_random(domain=a,random_type='normal',std=7,mean=3,dtype=itp)
                  out = fft.inverse_times(fft.times(inp))
                  assert_allclose(inp.val, out.val)

class Misc_Tests(unittest.TestCase):
    def test_RG_distance_1D(self):
        for dim1 in [10,11]:
            for zc1 in [False,True]:
                for d in [0.1,1,3.7]:
                    foo = RGSpace([dim1],zerocenter=zc1)
                    res = foo.get_distance_array('not')
                    assert_equal(res[zc1*(dim1//2)],0.)

    def test_RG_distance_2D(self):
      for dim1 in [10,11]:
        for dim2 in [9,28]:
          for zc1 in [False,True]:
            for zc2 in [False,True]:
              for d in [0.1,1,3.7]:
                foo = RGSpace([dim1,dim2],zerocenter=[zc1,zc2])
                res = foo.get_distance_array('not')
                assert_equal(res[zc1*(dim1//2),zc2*(dim2//2)],0.)

    def test_fft1D_numpy(self):
        _sub_fft1D("numpy")
    def test_fft1D_pyfftw(self):
        if 'pyfftw' not in di:
            raise SkipTest
        _sub_fft1D("fftw")

    def test_fft2D_numpy(self):
        _sub_fft2D("numpy")
    def test_fft2D_pyfftw(self):
        if 'pyfftw' not in di:
            raise SkipTest
        _sub_fft2D("fftw")

    def test_sht(self):
        if 'pyHealpix' not in di:
            raise SkipTest
        for lm in [0,3,6,11,30]:
            for tp in [np.float64,np.complex128,np.float32,np.complex64]:
                a = LMSpace(lmax=lm)
                b = LMGLTransformation.get_codomain(a)
                fft = FFTOperator(domain=a, target=b, domain_dtype=tp, target_dtype=tp)
                inp = Field.from_random(domain=a,random_type='normal',std=7,mean=3,dtype=tp)
                out = fft.inverse_times(fft.times(inp))
                assert_allclose(inp.val, out.val)

    def test_sht2(self):
        if 'pyHealpix' not in di:
            raise SkipTest
        for lm in [128,256]:
            for tp in [np.float64,np.complex128,np.float32,np.complex64]:
                a = LMSpace(lmax=lm)
                b = LMHPTransformation.get_codomain(a)
                fft = FFTOperator(domain=a, target=b, domain_dtype=tp, target_dtype=tp)
                inp = Field.from_random(domain=a,random_type='normal',std=1,mean=0,dtype=tp)
                out = fft.inverse_times(fft.times(inp))
                assert_allclose(inp.val, out.val,rtol=1e-3,atol=1e-3)
