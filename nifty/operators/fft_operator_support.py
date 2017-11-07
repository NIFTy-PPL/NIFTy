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
from .. import nifty_utilities as utilities
from ..low_level_library import hartley
from .. import dobj
from ..field import Field
from ..spaces.gl_space import GLSpace


class Transformation(object):
    def __init__(self, pdom, hdom, space):
        self.pdom = pdom
        self.hdom = hdom
        self.space = space


class RGRGTransformation(Transformation):
    def __init__(self, pdom, hdom, space):
        import pyfftw
        super(RGRGTransformation, self).__init__(pdom, hdom, space)
        pyfftw.interfaces.cache.enable()
        # correct for forward/inverse fft.
        # naively one would set power to 0.5 here in order to
        # apply effectively a factor of 1/sqrt(N) to the field.
        # BUT: the pixel volumes of the domain and codomain are different.
        # Hence, in order to produce the same scalar product, power==1.
        self.fct_p2h = pdom[space].scalar_dvol()
        self.fct_h2p = 1./(pdom[space].scalar_dvol()*hdom[space].dim)

    @property
    def unitary(self):
        return True

    def transform(self, x):
        """
        RG -> RG transform method.

        Parameters
        ----------
        x : Field
            The field to be transformed
        """
        axes = x.domain.axes[self.space]
        p2h = x.domain == self.pdom
        if dobj.dist_axis(x.val) in axes:
            raise NotImplementedError
        ldat = dobj.local_data(x.val)
        if p2h:
            Tval = Field(self.hdom, dobj.create_from_template(x.val,hartley(ldat, axes),dtype=x.val.dtype))
        else:
            Tval = Field(self.pdom, dobj.create_from_template(x.val,hartley(ldat, axes),dtype=x.val.dtype))
        fct = self.fct_p2h if p2h else self.fct_h2p
        if fct != 1:
            Tval *= fct

        return Tval


class SlicingTransformation(Transformation):
    def transform(self, x):
        axes = x.domain.axes[self.space]
        if dobj.dist_axis(x.val) in axes:
            raise NotImplementedError
        p2h = x.domain == self.pdom
        idat = dobj.local_data(x.val)
        if p2h:
            res = Field(self.hdom, dtype=x.dtype)
            odat = dobj.local_data(res.val)

            for slice in utilities.get_slice_list(idat.shape, axes):
                odat[slice] = self._slice_p2h(idat[slice])
        else:
            res = Field(self.pdom, dtype=x.dtype)
            odat = dobj.local_data(res.val)

            for slice in utilities.get_slice_list(idat.shape, axes):
                odat[slice] = self._slice_h2p(idat[slice])
        return res


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


class SphericalTransformation(SlicingTransformation):
    def __init__(self, pdom, hdom, space):
        super(SphericalTransformation, self).__init__(pdom, hdom, space)
        from pyHealpix import sharpjob_d

        self.lmax = self.hdom[self.space].lmax
        self.sjob = sharpjob_d()
        self.sjob.set_triangular_alm_info(self.hdom[self.space].lmax,
                                          self.hdom[self.space].mmax)
        if isinstance(self.pdom[self.space], GLSpace):
            self.sjob.set_Gauss_geometry(self.pdom[self.space].nlat,
                                         self.pdom[self.space].nlon)
        else:
            self.sjob.set_Healpix_geometry(self.pdom[self.space].nside)

    @property
    def unitary(self):
        return False

    def _slice_p2h(self, inp):
        rr = self.sjob.map2alm(inp)
        return buildIdx(rr, lmax=self.lmax)

    def _slice_h2p(self, inp):
        result = buildLm(inp, lmax=self.lmax)
        return self.sjob.alm2map(result)
