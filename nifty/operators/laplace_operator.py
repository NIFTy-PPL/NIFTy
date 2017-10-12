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
from ..field import Field
from ..spaces.power_space import PowerSpace
from .endomorphic_operator import EndomorphicOperator
from .. import DomainTuple
from .. import nifty_utilities as utilities


class LaplaceOperator(EndomorphicOperator):
    """An irregular LaplaceOperator with free boundary and excluding monopole.

    This LaplaceOperator implements the second derivative of a Field in
    PowerSpace on logarithmic or linear scale with vanishing curvature at the
    boundary, starting at the second entry of the Field. The second derivative
    of the Field on the irregular grid is calculated using finite differences.

    Parameters
    ----------
    logarithmic : boolean,
        Whether smoothness is calculated on a logarithmic scale or linear scale
        default : True
    space : int
        The index of the domain on which the operator acts
    """

    def __init__(self, domain, space=None, logarithmic=True):
        super(LaplaceOperator, self).__init__()
        self._domain = DomainTuple.make(domain)
        if space is None:
            if len(self._domain.domains) != 1:
                raise ValueError("need a Field with exactly one domain")
            space = 0
        space = int(space)
        if (space<0) or space>=len(self._domain.domains):
            raise ValueError("space index out of range")
        self._space = space

        if not isinstance(self._domain[self._space], PowerSpace):
            raise ValueError("Operator must act on a PowerSpace.")

        self._logarithmic = bool(logarithmic)

        pos = self.domain[self._space].k_lengths.copy()
        if self.logarithmic:
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

    @property
    def domain(self):
        return self._domain

    @property
    def unitary(self):
        return False

    @property
    def symmetric(self):
        return False

    @property
    def self_adjoint(self):
        return False

    @property
    def logarithmic(self):
        return self._logarithmic

    def _times(self, x):
        axes = x.domain.axes[self._space]
        axis = axes[0]
        nval = len(self._dposc)
        prefix = (slice(None),) * axis
        sl_l = prefix + (slice(None, -1),)  # "left" slice
        sl_r = prefix + (slice(1, None),)  # "right" slice
        dpos = self._dpos.reshape((1,)*axis + (nval-1,))
        dposc = self._dposc.reshape((1,)*axis + (nval,))
        deriv = (x.val[sl_r]-x.val[sl_l])/dpos  # defined between points
        ret = np.empty_like(x.val)
        ret[sl_l] = deriv
        ret[prefix + (-1,)] = 0.
        ret[sl_r] -= deriv
        ret /= np.sqrt(dposc)
        ret[prefix + (slice(None, 2),)] = 0.
        ret[prefix + (-1,)] = 0.
        return Field(self.domain, val=ret).weight(-0.5, spaces=self._space)

    def _adjoint_times(self, x):
        axes = x.domain.axes[self._space]
        axis = axes[0]
        nval = len(self._dposc)
        prefix = (slice(None),) * axis
        sl_l = prefix + (slice(None, -1),)  # "left" slice
        sl_r = prefix + (slice(1, None),)  # "right" slice
        dpos = self._dpos.reshape((1,)*axis + (nval-1,))
        dposc = self._dposc.reshape((1,)*axis + (nval,))
        y = x.weight(0.5, spaces=self._space).val
        y /= np.sqrt(dposc)
        y[prefix + (slice(None, 2),)] = 0.
        y[prefix + (-1,)] = 0.
        deriv = (y[sl_r]-y[sl_l])/dpos  # defined between points
        ret = np.empty_like(x.val)
        ret[sl_l] = deriv
        ret[prefix + (-1,)] = 0.
        ret[sl_r] -= deriv
        return Field(self.domain, val=ret).weight(-1, spaces=self._space)
