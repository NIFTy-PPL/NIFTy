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
from ..domains.rg_space import RGSpace
from ..field import Field
from .linear_operator import LinearOperator


class FieldZeroPadder(LinearOperator):
    def __init__(self, domain, new_shape, space=0):
        self._domain = DomainTuple.make(domain)
        self._space = utilities.infer_space(self._domain, space)
        dom = self._domain[self._space]
        if not isinstance(dom, RGSpace):
            raise TypeError("RGSpace required")
        if dom.harmonic:
            raise TypeError("RGSpace must not be harmonic")

        if len(new_shape) != len(dom.shape):
            raise ValueError("Shape mismatch")
        if any([a < b for a, b in zip(new_shape, dom.shape)]):
            raise ValueError("New shape must be larger than old shape")
        self._target = list(self._domain)
        self._target[self._space] = RGSpace(new_shape, dom.distances)
        self._target = DomainTuple.make(self._target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        v = x.val
        curshp = list(self._dom(mode).shape)
        tgtshp = self._tgt(mode).shape
        for d in self._target.axes[self._space]:
            idx = (slice(None),) * d

            v, x = dobj.ensure_not_distributed(v, (d,))

            if mode == self.TIMES:
                shp = list(x.shape)
                shp[d] = tgtshp[d]
                xnew = np.zeros(shp, dtype=x.dtype)
                xnew[idx + (slice(0, x.shape[d]),)] = x
            else:  # ADJOINT_TIMES
                xnew = x[idx + (slice(0, tgtshp[d]),)]

            curshp[d] = xnew.shape[d]
            v = dobj.from_local_data(curshp, xnew, distaxis=dobj.distaxis(v))
        return Field(self._tgt(mode), dobj.ensure_default_distributed(v))
