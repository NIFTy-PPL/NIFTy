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
from ..domains.dof_space import DOFSpace
from ..domains.power_space import PowerSpace
from ..field import Field
from ..utilities import infer_space, special_add_at
from .linear_operator import LinearOperator


class DOFDistributor(LinearOperator):
    """Operator which distributes actual degrees of freedom (dof) according to
    some distribution scheme into a higher dimensional space. This distribution
    scheme is defined by the dofdex, a degree of freedom index, which
    associates the entries within the operator's domain to locations in its
    target. This operator's domain is a DOFSpace, which is defined according to
    the distribution scheme.

    Parameters
    ----------
    dofdex: :class:`nifty8.field.Field` of integers
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
        if target is None:
            target = dofdex.domain
        self._target = DomainTuple.make(target)
        space = infer_space(self._target, space)
        partner = self._target[space]
        if not isinstance(dofdex, Field):
            raise TypeError("dofdex must be a Field")
        if not len(dofdex.domain) == 1:
            raise ValueError("dofdex must be defined on exactly one Space")
        if not np.issubdtype(dofdex.dtype, np.integer):
            raise TypeError("dofdex must contain integer numbers")
        if partner != dofdex.domain[0]:
            raise ValueError("incorrect dofdex domain")

        ldat = dofdex.val
        if ldat.size == 0:  # can happen for weird configurations
            nbin = 0
        else:
            nbin = ldat.max()
        nbin = nbin + 1
        if partner.scalar_dvol is not None:
            wgt = np.bincount(dofdex.val.ravel(), minlength=nbin)
            wgt = wgt*partner.scalar_dvol
        else:
            dvol = Field.from_raw(partner, partner.dvol).val
            wgt = np.bincount(dofdex.val.ravel(),
                              minlength=nbin, weights=dvol)
        # The explicit conversion to float64 is necessary because bincount
        # sometimes returns its result as an integer array, even when
        # floating-point weights are present ...
        wgt = wgt.astype(np.float64, copy=False)
        if (wgt == 0).any():
            raise ValueError("empty bins detected")

        self._init2(dofdex.val, space, DOFSpace(wgt))

    def _init2(self, dofdex, space, other_space):
        self._space = space
        dom = list(self._target)
        dom[self._space] = other_space
        self._domain = DomainTuple.make(dom)
        self._capability = self.TIMES | self.ADJOINT_TIMES

        self._dofdex = dofdex.ravel()
        firstaxis = self._target.axes[self._space][0]
        lastaxis = self._target.axes[self._space][-1]
        arrshape = self._target.shape
        presize = np.prod(arrshape[0:firstaxis], dtype=np.int64)
        postsize = np.prod(arrshape[lastaxis+1:], dtype=np.int64)
        self._hshape = (presize, self._domain[self._space].shape[0], postsize)
        self._pshape = (presize, self._dofdex.size, postsize)

    def _adjoint_times(self, x):
        arr = x.val
        arr = arr.reshape(self._pshape)
        oarr = np.zeros(self._hshape, dtype=x.dtype)
        oarr = special_add_at(oarr, 1, self._dofdex, arr)
        oarr = oarr.reshape(self._domain.shape)
        res = Field.from_raw(self._domain, oarr)
        return res

    def _times(self, x):
        arr = x.val
        arr = arr.reshape(self._hshape)
        oarr = np.empty(self._pshape, dtype=x.dtype)
        oarr[()] = arr[(slice(None), self._dofdex, slice(None))]
        return Field(self._target, oarr.reshape(self._target.shape))

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._times(x) if mode == self.TIMES else self._adjoint_times(x)


class PowerDistributor(DOFDistributor):
    """Operator which transforms between a PowerSpace and a harmonic domain.

    Parameters
    ----------
    target: Domain, tuple of Domain, or DomainTuple
        the total *target* domain of the operator.
    power_space: PowerSpace, optional
        the input sub-domain on which the operator acts.
        If not supplied, a matching PowerSpace with natural binbounds will be
        used.
    space: int, optional:
       The index of the sub-domain on which the operator acts.
       Can be omitted if `target` only has one sub-domain.
    """

    def __init__(self, target, power_space=None, space=None):
        # Initialize domain and target
        self._target = DomainTuple.make(target)
        self._space = infer_space(self._target, space)
        hspace = self._target[self._space]
        if not hspace.harmonic:
            raise ValueError("Operator requires harmonic target space")
        if power_space is None:
            power_space = PowerSpace(hspace)
        else:
            if not isinstance(power_space, PowerSpace):
                raise TypeError("power_space argument must be a PowerSpace")
            if power_space.harmonic_partner != hspace:
                raise ValueError("power_space does not match its partner")

        self._init2(power_space.pindex, self._space, power_space)
