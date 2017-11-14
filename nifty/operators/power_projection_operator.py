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

from .. import Field, DomainTuple
from ..spaces import PowerSpace
from .linear_operator import LinearOperator
from .. import dobj
import numpy as np


class PowerProjectionOperator(LinearOperator):
    def __init__(self, domain, power_space=None, space=None):
        super(PowerProjectionOperator, self).__init__()

        # Initialize domain and target
        self._domain = DomainTuple.make(domain)
        if space is None and len(self._domain) == 1:
            space = 0
        space = int(space)
        if space < 0 or space >= len(self.domain):
            raise ValueError("space index out of range")
        hspace = self._domain[space]
        if not hspace.harmonic:
            raise ValueError("Operator acts on harmonic spaces only")
        if power_space is None:
            power_space = PowerSpace(hspace)
        else:
            if not isinstance(power_space, PowerSpace):
                raise TypeError("power_space argument must be a PowerSpace")
            if power_space.harmonic_partner != hspace:
                raise ValueError("power_space does not match its partner")

        self._space = space
        tgt = list(self._domain)
        tgt[self._space] = power_space
        self._target = DomainTuple.make(tgt)

        pindex = self._target[self._space].pindex
        if dobj.default_distaxis() in self.domain.axes[self._space]:
            pindex = dobj.local_data(pindex)
        else:  # pindex must be available fully on every task
            pindex = dobj.to_global_data(pindex)
        self._pindex = pindex.ravel()
        firstaxis = self._domain.axes[self._space][0]
        lastaxis = self._domain.axes[self._space][-1]
        arrshape = dobj.local_shape(self._domain.shape, 0)
        presize = np.prod(arrshape[0:firstaxis], dtype=np.int)
        postsize = np.prod(arrshape[lastaxis+1:], dtype=np.int)
        self._hshape = (presize, self._target[self._space].shape[0], postsize)
        self._pshape = (presize, self._pindex.size, postsize)

    def _times(self, x):
        arr = dobj.local_data(x.weight(1).val)
        arr = arr.reshape(self._pshape)
        oarr = np.zeros(self._hshape, dtype=x.dtype)
        np.add.at(oarr, (slice(None), self._pindex, slice(None)), arr)
        if dobj.distaxis(x.val) in x.domain.axes[self._space]:
            oarr = dobj.np_allreduce_sum(oarr).reshape(self._target.shape)
            res = Field(self._target, dobj.from_global_data(oarr))
        else:
            oarr = oarr.reshape(dobj.local_shape(self._target.shape,
                                                 dobj.distaxis(x.val)))
            res = Field(self._target,
                        dobj.from_local_data(self._target.shape, oarr,
                                             dobj.default_distaxis()))
        return res.weight(-1, spaces=self._space)

    def _adjoint_times(self, x):
        res = Field.empty(self._domain, dtype=x.dtype)
        if dobj.distaxis(x.val) in x.domain.axes[self._space]:
            arr = dobj.to_global_data(x.val)
        else:
            arr = dobj.local_data(x.val)
        arr = arr.reshape(self._hshape)
        oarr = dobj.local_data(res.val).reshape(self._pshape)
        oarr[()] = arr[(slice(None), self._pindex, slice(None))]
        return res

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def unitary(self):
        return False
