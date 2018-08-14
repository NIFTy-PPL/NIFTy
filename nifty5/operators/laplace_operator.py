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
from ..field import Field
from ..utilities import infer_space
from .endomorphic_operator import EndomorphicOperator


class LaplaceOperator(EndomorphicOperator):
    """An irregular LaplaceOperator with free boundary and excluding monopole.

    This LaplaceOperator implements the second derivative of a Field in
    PowerSpace on logarithmic or linear scale with vanishing curvature at the
    boundary, starting at the second entry of the Field. The second derivative
    of the Field on the irregular grid is calculated using finite differences.

    Parameters
    ----------
    logarithmic : bool, optional
        Whether smoothness is calculated on a logarithmic scale or linear scale
        default : True
    space : int
        The index of the domain on which the operator acts
    """

    def __init__(self, domain, space=None, logarithmic=True):
        self._domain = DomainTuple.make(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._space = infer_space(self._domain, space)

        if not isinstance(self._domain[self._space], PowerSpace):
            raise ValueError("Operator must act on a PowerSpace.")

        pos = self.domain[self._space].k_lengths.copy()
        if logarithmic:
            pos[1:] = np.log(pos[1:])
            pos[0] = pos[1]-1.

        self._dpos = pos[1:]-pos[:-1]  # defined between points
        # centered distances (also has entries for the first and last point
        # for convenience, but they will never affect the result)
        self._dposc = np.empty_like(pos)
        self._dposc[:-1] = self._dpos
        self._dposc[-1] = 0.
        self._dposc[1:] += self._dpos
        self._dposc *= 0.5

    def apply(self, x, mode):
        self._check_input(x, mode)
        axes = x.domain.axes[self._space]
        axis = axes[0]
        nval = len(self._dposc)
        prefix = (slice(None),) * axis
        sl_l = prefix + (slice(None, -1),)  # "left" slice
        sl_r = prefix + (slice(1, None),)  # "right" slice
        dpos = self._dpos.reshape((1,)*axis + (nval-1,))
        dposc = self._dposc.reshape((1,)*axis + (nval,))
        locval = x.val
        if axis == dobj.distaxis(locval):
            locval = dobj.redistribute(locval, nodist=(axis,))
        val = dobj.local_data(locval)
        ret = np.empty_like(val)
        if mode == self.TIMES:
            deriv = (val[sl_r]-val[sl_l])/dpos  # defined between points
            ret[sl_l] = deriv
            ret[prefix + (-1,)] = 0.
            ret[sl_r] -= deriv
            ret /= dposc
            ret[prefix + (slice(None, 2),)] = 0.
            ret[prefix + (-1,)] = 0.
        else:
            val = val/dposc
            val[prefix + (slice(None, 2),)] = 0.
            val[prefix + (-1,)] = 0.
            deriv = (val[sl_r]-val[sl_l])/dpos  # defined between points
            ret[sl_l] = deriv
            ret[prefix + (-1,)] = 0.
            ret[sl_r] -= deriv
        ret = dobj.from_local_data(locval.shape, ret, dobj.distaxis(locval))
        if dobj.distaxis(locval) != dobj.distaxis(x.val):
            ret = dobj.redistribute(ret, dist=dobj.distaxis(x.val))
        return Field(self.domain, val=ret)
