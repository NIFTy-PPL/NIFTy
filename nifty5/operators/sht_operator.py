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

from __future__ import absolute_import, division, print_function

import numpy as np

from .. import dobj, utilities
from ..compat import *
from ..domain_tuple import DomainTuple
from ..domains.gl_space import GLSpace
from ..domains.lm_space import LMSpace
from ..field import Field
from .linear_operator import LinearOperator


class SHTOperator(LinearOperator):
    """Transforms between a harmonic domain on the sphere and a position
    domain counterpart.

    Built-in domain pairs are
      - an LMSpace and a HPSpace
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
        domain[space] must be a LMSpace.
    """

    def __init__(self, domain, target=None, space=None):
        super(SHTOperator, self).__init__()

        # Initialize domain and target
        self._domain = DomainTuple.make(domain)
        self._space = utilities.infer_space(self._domain, space)

        hspc = self._domain[self._space]
        if not isinstance(hspc, LMSpace):
            raise TypeError("SHTOperator only works on a LMSpace domain")
        if target is None:
            target = hspc.get_default_codomain()

        self._target = [dom for dom in self._domain]
        self._target[self._space] = target
        self._target = DomainTuple.make(self._target)
        hspc.check_codomain(target)
        target.check_codomain(hspc)

        from pyHealpix import sharpjob_d
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
            return (self._apply_spherical(x.real, mode) +
                    1j*self._apply_spherical(x.imag, mode))
        else:
            return self._apply_spherical(x, mode)

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
        tdom = self._tgt(mode)
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
