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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import division
import numpy as np
from ... import nifty_utilities as utilities
from ...low_level_library import hartley

class Transformation(object):
    def __init__(self, domain, codomain):
        self.domain = domain
        self.codomain = codomain

    def unitary(self):
        raise NotImplementedError

    def transform(self, val, axes=None):
        raise NotImplementedError


class RGRGTransformation(Transformation):
    def __init__(self, domain, codomain=None):
        import pyfftw
        super(RGRGTransformation, self).__init__(domain, codomain)
        pyfftw.interfaces.cache.enable()

    @property
    def unitary(self):
        return True

    def transform(self, val, axes=None):
        """
        RG -> RG transform method.

        Parameters
        ----------
        val : np.ndarray or distributed_data_object
            The value array which is to be transformed

        axes : None or tuple
            The axes along which the transformation should take place

        """
        # correct for forward/inverse fft.
        # naively one would set power to 0.5 here in order to
        # apply effectively a factor of 1/sqrt(N) to the field.
        # BUT: the pixel volumes of the domain and codomain are different.
        # Hence, in order to produce the same scalar product, power==1.
        if self.codomain.harmonic:
            fct = self.domain.scalar_dvol()
        else:
            fct = 1./(self.codomain.scalar_dvol()*self.domain.dim)

        # Perform the transformation
        if issubclass(val.dtype.type, np.complexfloating):
            Tval = hartley(val.real, axes) \
                  + 1j*hartley(val.imag, axes)
        else:
            Tval = hartley(val, axes)

        return Tval, fct


class SlicingTransformation(Transformation):
    def transform(self, val, axes=None):
        return_shape = np.array(val.shape)
        return_shape[list(axes)] = self.codomain.shape
        return_val = np.empty(tuple(return_shape), dtype=val.dtype)

        for slice in utilities.get_slice_list(val.shape, axes):
            return_val[slice] = self._transformation_of_slice(val[slice])
        return return_val, 1.

    def _transformation_of_slice(self, inp):
        raise NotImplementedError


def buildLm(nr, lmax):
    res = np.empty((len(nr)+lmax+1)//2, dtype=(nr[0]*1j).dtype)
    res[0:lmax+1] = nr[0:lmax+1]
    res[lmax+1:] = np.sqrt(0.5)*(nr[lmax+1::2] + 1j*nr[lmax+2::2])
    return res


def buildIdx(nr, lmax):
    res = np.empty((lmax+1)*(lmax+1), dtype=nr[0].real.dtype)
    res[0:lmax+1] = nr[0:lmax+1].real
    res[lmax+1::2] = np.sqrt(2)*nr[lmax+1:].real
    res[lmax+2::2] = np.sqrt(2)*nr[lmax+1:].imag
    return res


class HPLMTransformation(SlicingTransformation):
    @property
    def unitary(self):
        return False

    def _transformation_of_slice(self, inp):
        from pyHealpix import map2alm

        lmax = self.codomain.lmax
        mmax = lmax

        if issubclass(inp.dtype.type, np.complexfloating):
            rr = map2alm(inp.real, lmax, mmax)
            rr = buildIdx(rr, lmax=lmax)
            ri = map2alm(inp.imag, lmax, mmax)
            ri = buildIdx(ri, lmax=lmax)
            return rr + 1j*ri
        else:
            rr = map2alm(inp, lmax, mmax)
            return buildIdx(rr, lmax=lmax)


class LMHPTransformation(SlicingTransformation):
    @property
    def unitary(self):
        return False

    def _transformation_of_slice(self, inp):
        from pyHealpix import alm2map

        nside = self.codomain.nside
        lmax = self.domain.lmax
        mmax = lmax

        if issubclass(inp.dtype.type, np.complexfloating):
            rr = buildLm(inp.real, lmax=lmax)
            ri = buildLm(inp.imag, lmax=lmax)
            rr = alm2map(rr, lmax, mmax, nside)
            ri = alm2map(ri, lmax, mmax, nside)
            return rr + 1j*ri
        else:
            rr = buildLm(inp, lmax=lmax)
            return alm2map(rr, lmax, mmax, nside)


class GLLMTransformation(SlicingTransformation):
    @property
    def unitary(self):
        return False

    def _transformation_of_slice(self, inp):
        from pyHealpix import sharpjob_d

        lmax = self.codomain.lmax
        mmax = self.codomain.mmax

        sjob = sharpjob_d()
        sjob.set_Gauss_geometry(self.domain.nlat, self.domain.nlon)
        sjob.set_triangular_alm_info(lmax, mmax)
        if issubclass(inp.dtype.type, np.complexfloating):
            rr = sjob.map2alm(inp.real)
            rr = buildIdx(rr, lmax=lmax)
            ri = sjob.map2alm(inp.imag)
            ri = buildIdx(ri, lmax=lmax)
            return rr + 1j*ri
        else:
            rr = sjob.map2alm(inp)
            return buildIdx(rr, lmax=lmax)


class LMGLTransformation(SlicingTransformation):
    @property
    def unitary(self):
        return False

    def _transformation_of_slice(self, inp):
        from pyHealpix import sharpjob_d

        lmax = self.domain.lmax
        mmax = self.domain.mmax

        sjob = sharpjob_d()
        sjob.set_Gauss_geometry(self.codomain.nlat, self.codomain.nlon)
        sjob.set_triangular_alm_info(lmax, mmax)
        if issubclass(inp.dtype.type, np.complexfloating):
            rr = buildLm(inp.real, lmax=lmax)
            ri = buildLm(inp.imag, lmax=lmax)
            return sjob.alm2map(rr) + 1j*sjob.alm2map(ri)
        else:
            result = buildLm(inp, lmax=lmax)
            return sjob.alm2map(result)
