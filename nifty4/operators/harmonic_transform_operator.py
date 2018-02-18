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

import numpy as np
from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from .linear_operator import LinearOperator
from .. import dobj
from .. import utilities
from ..field import Field
from ..domains.gl_space import GLSpace


class HarmonicTransformOperator(LinearOperator):
    """Transforms between a harmonic domain and a position domain counterpart.

    Built-in domain pairs are

      - a harmonic and a non-harmonic RGSpace (with matching distances)
      - an LMSpace and a LMSpace
      - an LMSpace and a GLSpace

    The supported operations are times() and adjoint_times().

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        The domain of the data that is input by "times" and output by
        "adjoint_times".
    target : Domain, optional
        The target domain of the transform operation.
        If omitted, a domain will be chosen automatically.
        Whenever the input domain of the transform is an RGSpace, the codomain
        (and its parameters) are uniquely determined.
        For LMSpace, a GLSpace of sufficient resolution is chosen.
    space : int, optional
        The index of the domain on which the operator should act
        If None, it is set to 0 if domain contains exactly one subdomain.
        domain[space] must be a harmonic domain.
    """

    def __init__(self, domain, target=None, space=None):
        super(HarmonicTransformOperator, self).__init__()

        # Initialize domain and target
        self._domain = DomainTuple.make(domain)
        self._space = utilities.infer_space(self._domain, space)

        hspc = self._domain[self._space]
        if not hspc.harmonic:
            raise TypeError(
                "HarmonicTransformOperator only works on a harmonic space")
        if target is None:
            target = hspc.get_default_codomain()

        self._target = [dom for dom in self._domain]
        self._target[self._space] = target
        self._target = DomainTuple.make(self._target)
        hspc.check_codomain(target)
        target.check_codomain(hspc)

        if isinstance(hspc, RGSpace):
            self._applyfunc = self._apply_cartesian
            import pyfftw
            pyfftw.interfaces.cache.enable()
        else:
            from pyHealpix import sharpjob_d
            self._applyfunc = self._apply_spherical
            self.lmax = hspc.lmax
            self.mmax = hspc.mmax
            self.sjob = sharpjob_d()
            self.sjob.set_triangular_alm_info(self.lmax, self.mmax)
            if isinstance(target, GLSpace):
                self.sjob.set_Gauss_geometry(target.nlat, target.nlon)
            else:
                self.sjob.set_Healpix_geometry(target.nside)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if np.issubdtype(x.dtype, np.complexfloating):
            return (self._applyfunc(x.real, mode) +
                    1j*self._applyfunc(x.imag, mode))
        else:
            return self._applyfunc(x, mode)

    def _apply_cartesian(self, x, mode):
        from pyfftw.interfaces.numpy_fft import fftn
        axes = x.domain.axes[self._space]
        tdom = self._target if x.domain == self._domain else self._domain
        oldax = dobj.distaxis(x.val)
        if oldax not in axes:  # straightforward, no redistribution needed
            ldat = x.local_data
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
            rem_axes = tuple(i for i in axes if i != oldax)
            tmp = x.val
            ldat = dobj.local_data(tmp)
            ldat = utilities.my_fftn_r2c(ldat, axes=rem_axes)
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
        Tval = Field(tdom, tmp)
        fct = self._domain[self._space].scalar_dvol
        if fct != 1:
            Tval *= fct

        return Tval

    def _slice_p2h(self, inp):
        rr = self.sjob.alm2map_adjoint(inp)
        if len(rr) != ((self.mmax+1)*(self.mmax+2))//2 + \
                      (self.mmax+1)*(self.lmax-self.mmax):
            raise ValueError("array length mismatch")
        res = np.empty(2*len(rr)-self.lmax-1, dtype=rr[0].real.dtype)
        res[0:self.lmax+1] = rr[0:self.lmax+1].real
        res[self.lmax+1::2] = np.sqrt(2)*rr[self.lmax+1:].real
        res[self.lmax+2::2] = np.sqrt(2)*rr[self.lmax+1:].imag
        return res/np.sqrt(np.pi*4)

    def _slice_h2p(self, inp):
        res = np.empty((len(inp)+self.lmax+1)//2, dtype=(inp[0]*1j).dtype)
        if len(res) != ((self.mmax+1)*(self.mmax+2))//2 + \
                       (self.mmax+1)*(self.lmax-self.mmax):
            raise ValueError("array length mismatch")
        res[0:self.lmax+1] = inp[0:self.lmax+1]
        res[self.lmax+1:] = np.sqrt(0.5)*(inp[self.lmax+1::2] +
                                          1j*inp[self.lmax+2::2])
        res = self.sjob.alm2map(res)
        return res/np.sqrt(np.pi*4)

    def _apply_spherical(self, x, mode):
        axes = x.domain.axes[self._space]
        axis = axes[0]
        tval = x.val
        if dobj.distaxis(tval) == axis:
            tval = dobj.redistribute(tval, nodist=(axis,))
        distaxis = dobj.distaxis(tval)

        p2h = not x.domain[self._space].harmonic
        tdom = self._target if x.domain == self._domain else self._domain
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

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
