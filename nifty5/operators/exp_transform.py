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
from ..domains.power_space import PowerSpace
from ..domains.rg_space import RGSpace
from ..field import Field
from ..utilities import infer_space, special_add_at
from .linear_operator import LinearOperator


class ExpTransform(LinearOperator):
    """
    Transforms log-space to target.

    This operator creates a log-space subject to the degrees of freedom and
    and its target-domain.
    Then transforms between this log-space and its target, which lives in
    normal units.

    E.g: A field in log-log-space can be transformed into log-norm-space,
         that is the y-axis stays logarithmic, but the x-axis is transfromed.

    Parameters
    ----------
    target : domain, tuple of domains or DomainTuple
        The full output domain
    dof : int
        The degrees of freedom of the log-domain, i.e. the number of bins.
    """
    def __init__(self, target, dof, space=0):
        self._target = DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._space = infer_space(self._target, space)
        tgt = self._target[self._space]
        if not ((isinstance(tgt, RGSpace) and tgt.harmonic) or
                isinstance(tgt, PowerSpace)):
            raise ValueError(
                "Target must be a harmonic RGSpace or a power space.")

        if np.isscalar(dof):
            dof = np.full(len(tgt.shape), int(dof), dtype=np.int)
        dof = np.array(dof)
        ndim = len(tgt.shape)

        t_mins = np.empty(ndim)
        bindistances = np.empty(ndim)
        self._bindex = [None] * ndim
        self._frac = [None] * ndim

        for i in range(ndim):
            if isinstance(tgt, RGSpace):
                rng = np.arange(tgt.shape[i])
                tmp = np.minimum(rng, tgt.shape[i]+1-rng)
                k_array = tmp * tgt.distances[i]
            else:
                k_array = tgt.k_lengths

            # avoid taking log of first entry
            log_k_array = np.log(k_array[1:])

            # Interpolate log_k_array linearly
            t_max = np.max(log_k_array)
            t_min = np.min(log_k_array)

            # Save t_min for later
            t_mins[i] = t_min

            bindistances[i] = (t_max-t_min) / (dof[i]-1)
            coord = np.append(0., 1. + (log_k_array-t_min) / bindistances[i])
            self._bindex[i] = np.floor(coord).astype(int)

            # Interpolated value is computed via
            # (1.-frac)*<value from this bin> + frac*<value from next bin>
            # 0 <= frac < 1.
            self._frac[i] = coord - self._bindex[i]

        from ..domains.log_rg_space import LogRGSpace
        log_space = LogRGSpace(2*dof+1, bindistances,
                               t_mins, harmonic=False)
        self._domain = [dom for dom in self._target]
        self._domain[self._space] = log_space
        self._domain = DomainTuple.make(self._domain)

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

            v, x = dobj.ensure_not_distributed(v, (d,))

            if mode == self.ADJOINT_TIMES:
                shp = list(x.shape)
                shp[d] = tgtshp[d]
                xnew = np.zeros(shp, dtype=x.dtype)
                xnew = special_add_at(xnew, d, self._bindex[d-d0], x*(1.-wgt))
                xnew = special_add_at(xnew, d, self._bindex[d-d0]+1, x*wgt)
            else:  # TIMES
                xnew = x[idx + (self._bindex[d-d0],)] * (1.-wgt)
                xnew += x[idx + (self._bindex[d-d0]+1,)] * wgt

            curshp[d] = xnew.shape[d]
            v = dobj.from_local_data(curshp, xnew, distaxis=dobj.distaxis(v))
        return Field(self._tgt(mode), dobj.ensure_default_distributed(v))
