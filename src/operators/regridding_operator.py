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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..field import Field
from ..utilities import infer_space, special_add_at
from .linear_operator import LinearOperator


class RegriddingOperator(LinearOperator):
    """Linearly interpolates an RGSpace to an RGSpace with coarser resolution.

    Parameters
    ----------
    domain : Domain, DomainTuple or tuple of Domain
        domain[space] needs to be an :class:`RGSpace`.
    new_shape : tuple of int
        Shape of the space which domain[space] is replaced by. Each entry must
        be smaller or equal to the respective entry in `domain[space].shape`.
    space : int
        Index of space in `domain` on which the operator shall act.
        Default is 0.
    """
    def __init__(self, domain, new_shape, space=0):
        self._domain = DomainTuple.make(domain)
        self._space = infer_space(self._domain, space)
        dom = self._domain[self._space]

        if not isinstance(dom, RGSpace):
            raise TypeError("RGSpace required")
        if len(new_shape) != len(dom.shape):
            raise ValueError("Shape mismatch")
        if any([a > b for a, b in zip(new_shape, dom.shape)]):
            raise ValueError("New shape must not be larger than old shape")
        if any([ii <= 0 for ii in new_shape]):
            raise ValueError('New shape must not be zero or negative.')

        newdist = tuple(dom.distances[i]*dom.shape[i]/new_shape[i]
                        for i in range(len(dom.shape)))

        tgt = RGSpace(new_shape, newdist)
        self._target = list(self._domain)
        self._target[self._space] = tgt
        self._target = DomainTuple.make(self._target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

        ndim = len(new_shape)
        self._bindex = [None] * ndim
        self._frac = [None] * ndim
        for d in range(ndim):
            tmp = np.arange(new_shape[d])*(newdist[d]/dom.distances[d])
            self._bindex[d] = np.minimum(dom.shape[d]-2, tmp.astype(np.int64))
            self._frac[d] = tmp-self._bindex[d]

    def apply(self, x, mode):
        self._check_input(x, mode)
        v = x.val
        ndim = len(self.target.shape)
        curshp = list(self._dom(mode).shape)
        tgtshp = self._tgt(mode).shape
        d0 = self._target.axes[self._space][0]
        for d in self._target.axes[self._space]:
            idx = (slice(None),) * d
            wgt = self._frac[d-d0].reshape((1,)*d + (-1,) + (1,)*(ndim-d-1))

            if mode == self.ADJOINT_TIMES:
                shp = list(v.shape)
                shp[d] = tgtshp[d]
                xnew = np.zeros(shp, dtype=v.dtype)
                xnew = special_add_at(xnew, d, self._bindex[d-d0], v*(1.-wgt))
                xnew = special_add_at(xnew, d, self._bindex[d-d0]+1, v*wgt)
            else:  # TIMES
                xnew = v[idx + (self._bindex[d-d0],)] * (1.-wgt)
                xnew += v[idx + (self._bindex[d-d0]+1,)] * wgt

            curshp[d] = xnew.shape[d]
            v = xnew
        return Field(self._tgt(mode), xnew)
