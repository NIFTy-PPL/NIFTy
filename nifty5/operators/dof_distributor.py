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

from .. import dobj
from ..compat import *
from ..domain_tuple import DomainTuple
from ..domains.dof_space import DOFSpace
from ..field import Field
from ..utilities import infer_space, special_add_at
from .linear_operator import LinearOperator


class DOFDistributor(LinearOperator):
    """Operator which distributes actual degrees of freedom (dof) according to
    some distribution scheme into a higher dimensional space. This distribution
    scheme is defined by the dofdex, a degree of freedom index, which
    associates the entries within the operators domain to locations in its
    target. This operator's domain is a DOFSpace, which is defined according to
    the distribution scheme.

    Parameters
    ----------
    dofdex: Field of integers
        An integer Field on exactly one Space establishing the association
        between the locations in the operator's target and the underlying
        degrees of freedom in its domain.
        It has to start at 0 and it increases monotonically, no empty bins are
        allowed.
    target: Domain, tuple of Domain, or DomainTuple, optional
        The target of the operator. If not specified, the domain of the dofdex
        is used.
    space: int, optional:
       The index of the sub-domain on which the operator acts.
       Can be omitted if `target` only has one sub-domain.
    """

    def __init__(self, dofdex, target=None, space=None):
        super(DOFDistributor, self).__init__()

        if target is None:
            target = dofdex.domain
        self._target = DomainTuple.make(target)
        space = infer_space(self._target, space)
        partner = self._target[space]
        if not isinstance(dofdex, Field):
            raise TypeError("dofdex must be a Field")
        if not len(dofdex.domain) == 1:
            raise ValueError("dofdex must live on exactly one Space")
        if not np.issubdtype(dofdex.dtype, np.integer):
            raise TypeError("dofdex must contain integer numbers")
        if partner != dofdex.domain[0]:
            raise ValueError("incorrect dofdex domain")

        ldat = dofdex.local_data
        if ldat.size == 0:  # can happen for weird configurations
            nbin = 0
        else:
            nbin = ldat.max()
        nbin = dobj.np_allreduce_max(np.array(nbin))[()] + 1
        if partner.scalar_dvol is not None:
            wgt = np.bincount(dofdex.local_data.ravel(), minlength=nbin)
            wgt = wgt*partner.scalar_dvol
        else:
            dvol = Field.from_global_data(partner, partner.dvol).local_data
            wgt = np.bincount(dofdex.local_data.ravel(),
                              minlength=nbin, weights=dvol)
        # The explicit conversion to float64 is necessary because bincount
        # sometimes returns its result as an integer array, even when
        # floating-point weights are present ...
        wgt = wgt.astype(np.float64, copy=False)
        wgt = dobj.np_allreduce_sum(wgt)
        if (wgt == 0).any():
            raise ValueError("empty bins detected")

        self._init2(dofdex.val, space, DOFSpace(wgt))

    def _init2(self, dofdex, space, other_space):
        self._space = space
        dom = list(self._target)
        dom[self._space] = other_space
        self._domain = DomainTuple.make(dom)

        if dobj.default_distaxis() in self._domain.axes[self._space]:
            dofdex = dobj.local_data(dofdex)
        else:  # dofdex must be available fully on every task
            dofdex = dobj.to_global_data(dofdex)
        self._dofdex = dofdex.ravel()
        firstaxis = self._target.axes[self._space][0]
        lastaxis = self._target.axes[self._space][-1]
        arrshape = dobj.local_shape(self._target.shape, 0)
        presize = np.prod(arrshape[0:firstaxis], dtype=np.int)
        postsize = np.prod(arrshape[lastaxis+1:], dtype=np.int)
        self._hshape = (presize, self._domain[self._space].shape[0], postsize)
        self._pshape = (presize, self._dofdex.size, postsize)

    def _adjoint_times(self, x):
        arr = x.local_data
        arr = arr.reshape(self._pshape)
        oarr = np.zeros(self._hshape, dtype=x.dtype)
        oarr = special_add_at(arr, 1, self._dofdex, oarr)
        if dobj.distaxis(x.val) in x.domain.axes[self._space]:
            oarr = dobj.np_allreduce_sum(oarr).reshape(self._domain.shape)
            res = Field.from_global_data(self._domain, oarr)
        else:
            oarr = oarr.reshape(dobj.local_shape(self._domain.shape,
                                                 dobj.distaxis(x.val)))
            res = Field(self._domain,
                        dobj.from_local_data(self._domain.shape, oarr,
                                             dobj.default_distaxis()))
        return res

    def _times(self, x):
        if dobj.distaxis(x.val) in x.domain.axes[self._space]:
            arr = x.to_global_data()
        else:
            arr = x.local_data
        arr = arr.reshape(self._hshape)
        oarr = np.empty(self._pshape, dtype=x.dtype)
        oarr[()] = arr[(slice(None), self._dofdex, slice(None))]
        return Field.from_local_data(
            self._target, oarr.reshape(self._target.local_shape))

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._times(x) if mode == self.TIMES else self._adjoint_times(x)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
