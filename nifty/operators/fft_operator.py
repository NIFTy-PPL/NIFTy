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

import numpy as np
from .. import DomainTuple
from ..spaces import RGSpace
from .linear_operator import LinearOperator
from .. import dobj
from .. import utilities
from ..field import Field
from ..spaces.gl_space import GLSpace


class FFTOperator(LinearOperator):
    """Transforms between a pair of position and harmonic domains.

    Built-in domain pairs are
      - a harmonic and a non-harmonic RGSpace (with matching distances)
      - a HPSpace and a LMSpace
      - a GLSpace and a LMSpace
    Within a domain pair, both orderings are possible.

    For RGSpaces, the operator provides the full set of operations.
    For the sphere-related domains, it only supports the transform from
    harmonic to position space and its adjoint; if the operator domain is
    harmonic, this will be times() and adjoint_times(), otherwise
    inverse_times() and adjoint_inverse_times()

    Parameters
    ----------
    domain: Space or single-element tuple of Spaces
        The domain of the data that is input by "times" and output by
        "adjoint_times".
    space: the index of the space on which the operator should act
        If None, it is set to 0 if domain contains exactly one space
    target: Space or single-element tuple of Spaces (optional)
        The domain of the data that is output by "times" and input by
        "adjoint_times".
        If omitted, a co-domain will be chosen automatically.
        Whenever "domain" is an RGSpace, the codomain (and its parameters) are
        uniquely determined.
        For GLSpace, HPSpace, and LMSpace, a sensible (but not unique)
        co-domain is chosen that should work satisfactorily in most situations,
        but for full control, the user should explicitly specify a codomain.
    """

    def __init__(self, domain, target=None, space=None):
        super(FFTOperator, self).__init__()

        # Initialize domain and target
        self._domain = DomainTuple.make(domain)
        self._space = utilities.infer_space(self._domain, space)

        adom = self._domain[self._space]
        if target is None:
            target = adom.get_default_codomain()

        self._target = [dom for dom in self._domain]
        self._target[self._space] = target
        self._target = DomainTuple.make(self._target)
        adom.check_codomain(target)
        target.check_codomain(adom)

        if isinstance(adom, RGSpace):
            self._applyfunc = self._apply_cartesian
            self._capability = self._all_ops
            import pyfftw
            pyfftw.interfaces.cache.enable()
        else:
            from pyHealpix import sharpjob_d
            self._applyfunc = self._apply_spherical
            hspc = adom if adom.harmonic else target
            pspc = target if adom.harmonic else adom
            self.lmax = hspc.lmax
            self.mmax = hspc.mmax
            self.sjob = sharpjob_d()
            self.sjob.set_triangular_alm_info(self.lmax, self.mmax)
            if isinstance(pspc, GLSpace):
                self.sjob.set_Gauss_geometry(pspc.nlat, pspc.nlon)
            else:
                self.sjob.set_Healpix_geometry(pspc.nside)
            if adom.harmonic:
                self._capability = self.TIMES | self.ADJOINT_TIMES
            else:
                self._capability = (self.INVERSE_TIMES |
                                    self.INVERSE_ADJOINT_TIMES)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if np.issubdtype(x.dtype, np.complexfloating):
            return (self._applyfunc(x.real, mode) +
                    1j*self._applyfunc(x.imag, mode))
        else:
            return self._applyfunc(x, mode)

    def _apply_cartesian(self, x, mode):
        """
        RG -> RG transform method.

        Parameters
        ----------
        x : Field
            The field to be transformed
        """
        from pyfftw.interfaces.numpy_fft import fftn
        axes = x.domain.axes[self._space]
        tdom = self._target if x.domain == self._domain else self._domain
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
        if mode & (LinearOperator.TIMES | LinearOperator.ADJOINT_TIMES):
            fct = self._domain[self._space].scalar_dvol()
        else:
            fct = self._target[self._space].scalar_dvol()
        if fct != 1:
            Tval *= fct

        return Tval

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
        return self._capability
