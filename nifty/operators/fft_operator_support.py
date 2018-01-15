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
from .. import utilities
from .. import dobj
from ..field import Field
from ..spaces.gl_space import GLSpace
from .linear_operator import LinearOperator


class Transformation(object):
    def __init__(self, hdom, pdom, space):
        self.hdom = hdom
        self.pdom = pdom
        self.space = space


class RGRGTransformation(Transformation):
    def __init__(self, hdom, pdom, space):
        import pyfftw
        super(RGRGTransformation, self).__init__(hdom, pdom, space)
        pyfftw.interfaces.cache.enable()
        self.fct_noninverse = hdom[space].scalar_dvol()
        self.fct_inverse = 1./(hdom[space].scalar_dvol()*hdom[space].dim)

    @property
    def unitary(self):
        return True

    def apply(self, x, mode):
        """
        RG -> RG transform method.

        Parameters
        ----------
        x : Field
            The field to be transformed
        """
        from pyfftw.interfaces.numpy_fft import fftn
        axes = x.domain.axes[self.space]
        p2h = x.domain == self.pdom
        tdom = self.hdom if p2h else self.pdom
        oldax = dobj.distaxis(x.val)
        if oldax not in axes:  # straightforward, no redistribution needed
            ldat = dobj.local_data(x.val)
            ldat = utilities.hartley(ldat, axes=axes)
            tmp = dobj.from_local_data(x.val.shape, ldat, distaxis=oldax)
        elif len(axes) < len(x.shape) or len(axes) == 1:
            # we can use one Hartley pass in between the redistributions
            tmp = dobj.redistribute(x.val, nodist=axes)
            newax = dobj.distaxis(tmp)
            ldat = dobj.local_data(tmp)
            ldat = utilities.hartley(ldat, axes=axes)
            tmp = dobj.from_local_data(tmp.shape, ldat, distaxis=newax)
            tmp = dobj.redistribute(tmp, dist=oldax)
        else:  # two separate, full FFTs needed
            # ideal strategy for the moment would be:
            # - do real-to-complex FFT on all local axes
            # - fill up array
            # - redistribute array
            # - do complex-to-complex FFT on remaining axis
            # - add re+im
            # - redistribute back
            if True:
                rem_axes = tuple(i for i in axes if i != oldax)
                tmp = x.val
                ldat = dobj.local_data(tmp)
                ldat = utilities.my_fftn_r2c(ldat, axes=rem_axes)
                # new, experimental code
                if True:
                    if oldax != 0:
                        raise ValueError("bad distribution")
                    ldat2 = ldat.reshape((ldat.shape[0],
                                          np.prod(ldat.shape[1:])))
                    shp2d = (x.val.shape[0], np.prod(x.val.shape[1:]))
                    tmp = dobj.from_local_data(shp2d, ldat2, distaxis=0)
                    tmp = dobj.transpose(tmp)
                    ldat2 = dobj.local_data(tmp)
                    ldat2 = fftn(ldat2, axes=(1,))
                    ldat2 = ldat2.real+ldat2.imag
                    tmp = dobj.from_local_data(tmp.shape, ldat2, distaxis=0)
                    tmp = dobj.transpose(tmp)
                    ldat2 = dobj.local_data(tmp).reshape(ldat.shape)
                    tmp = dobj.from_local_data(x.val.shape, ldat2, distaxis=0)
                else:
                    tmp = dobj.from_local_data(tmp.shape, ldat, distaxis=oldax)
                    tmp = dobj.redistribute(tmp, nodist=(oldax,))
                    newax = dobj.distaxis(tmp)
                    ldat = dobj.local_data(tmp)
                    ldat = fftn(ldat, axes=(oldax,))
                    ldat = ldat.real+ldat.imag
                    tmp = dobj.from_local_data(tmp.shape, ldat, distaxis=newax)
                    tmp = dobj.redistribute(tmp, dist=oldax)
            else:
                tmp = dobj.redistribute(x.val, nodist=(oldax,))
                newax = dobj.distaxis(tmp)
                ldat = dobj.local_data(tmp)
                ldat = fftn(ldat, axes=(oldax,))
                tmp = dobj.from_local_data(tmp.shape, ldat, distaxis=newax)
                tmp = dobj.redistribute(tmp, dist=oldax)
                rem_axes = tuple(i for i in axes if i != oldax)
                ldat = dobj.local_data(tmp)
                ldat = fftn(ldat, axes=rem_axes)
                ldat = ldat.real+ldat.imag
                tmp = dobj.from_local_data(tmp.shape, ldat, distaxis=oldax)
        Tval = Field(tdom, tmp)
        if (mode == LinearOperator.TIMES or
                mode == LinearOperator.ADJOINT_TIMES):
            fct = self.fct_noninverse
        else:
            fct = self.fct_inverse
        if fct != 1:
            Tval *= fct

        return Tval


class SphericalTransformation(Transformation):
    def __init__(self, hdom, pdom, space):
        super(SphericalTransformation, self).__init__(hdom, pdom, space)
        from pyHealpix import sharpjob_d

        self.lmax = self.hdom[self.space].lmax
        self.mmax = self.hdom[self.space].mmax
        self.sjob = sharpjob_d()
        self.sjob.set_triangular_alm_info(self.lmax, self.mmax)
        if isinstance(self.pdom[self.space], GLSpace):
            self.sjob.set_Gauss_geometry(self.pdom[self.space].nlat,
                                         self.pdom[self.space].nlon)
        else:
            self.sjob.set_Healpix_geometry(self.pdom[self.space].nside)

    @property
    def unitary(self):
        return False

    def _slice_p2h(self, inp):
        rr = self.sjob.alm2map_adjoint(inp)
        assert len(rr) == ((self.mmax+1)*(self.mmax+2))//2 + \
                          (self.mmax+1)*(self.lmax-self.mmax)
        res = np.empty(2*len(rr)-self.lmax-1, dtype=rr[0].real.dtype)
        res[0:self.lmax+1] = rr[0:self.lmax+1].real
        res[self.lmax+1::2] = np.sqrt(2)*rr[self.lmax+1:].real
        res[self.lmax+2::2] = np.sqrt(2)*rr[self.lmax+1:].imag
        return res/np.sqrt(np.pi*4)

    def _slice_h2p(self, inp):
        res = np.empty((len(inp)+self.lmax+1)//2, dtype=(inp[0]*1j).dtype)
        assert len(res) == ((self.mmax+1)*(self.mmax+2))//2 + \
                           (self.mmax+1)*(self.lmax-self.mmax)
        res[0:self.lmax+1] = inp[0:self.lmax+1]
        res[self.lmax+1:] = np.sqrt(0.5)*(inp[self.lmax+1::2] +
                                          1j*inp[self.lmax+2::2])
        res = self.sjob.alm2map(res)
        return res/np.sqrt(np.pi*4)

    def apply(self, x, mode):
        axes = x.domain.axes[self.space]
        axis = axes[0]
        tval = x.val
        if dobj.distaxis(tval) == axis:
            tval = dobj.redistribute(tval, nodist=(axis,))
        distaxis = dobj.distaxis(tval)

        p2h = x.domain == self.pdom
        tdom = self.hdom if p2h else self.pdom
        func = self._slice_p2h if p2h else self._slice_h2p
        idat = dobj.local_data(tval)
        odat = np.empty(dobj.local_shape(tdom.shape, distaxis=distaxis),
                        dtype=x.dtype)
        for slice in utilities.get_slice_list(idat.shape, axes):
            odat[slice] = func(idat[slice])
        odat = dobj.from_local_data(tdom.shape, odat, distaxis)
        if distaxis != dobj.distaxis(x.val):
            odat = dobj.redistribute(odat, dist=dobj.distaxis(x.val))
        return Field(tdom, odat)
