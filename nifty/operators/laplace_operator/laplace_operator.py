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
from ...field import Field
from ...spaces.power_space import PowerSpace
from ..endomorphic_operator import EndomorphicOperator
from ... import sqrt
from ... import nifty_utilities as utilities


class LaplaceOperator(EndomorphicOperator):
    """A irregular LaplaceOperator with free boundary and excluding monopole.

    This LaplaceOperator implements the second derivative of a Field in
    PowerSpace  on logarithmic or linear scale with vanishing curvature at the
    boundary, starting at the second entry of the Field. The second derivative
    of the Field on the irregular grid is calculated using finite differences.

    Parameters
    ----------
    logarithmic : boolean,
        Whether smoothness is calculated on a logarithmic scale or linear scale
        default : True
    """

    def __init__(self, domain, default_spaces=None, logarithmic=True):
        super(LaplaceOperator, self).__init__(default_spaces)
        self._domain = self._parse_domain(domain)
        if len(self.domain) != 1:
            raise ValueError("The domain must contain exactly one PowerSpace.")

        if not isinstance(self.domain[0], PowerSpace):
            raise TypeError("The domain must contain exactly one PowerSpace.")

        self._logarithmic = bool(logarithmic)

        pos = self.domain[0].kindex.copy()
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

    def _add_attributes_to_copy(self, copy, **kwargs):
        copy._domain = self._domain
        copy._logarithmic = self._logarithmic
        copy._dpos = self._dpos
        copy._dposc = self._dposc
        copy = super(LaplaceOperator, self)._add_attributes_to_copy(copy,
                                                                    **kwargs)
        return copy

    @property
    def target(self):
        return self._domain

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

    def _times(self, x, spaces):
        spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))
        if spaces is None:
            # this case means that x lives on only one space, which is
            # identical to the space in the domain of `self`. Otherwise the
            # input check of LinearOperator would have failed.
            axes = x.domain_axes[0]
        else:
            axes = x.domain_axes[spaces[0]]
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
        ret /= sqrt(dposc)
        ret[prefix + (slice(None, 2),)] = 0.
        ret[prefix + (-1,)] = 0.
        return Field(self.domain, val=ret).weight(power=-0.5, spaces=spaces)

    def _adjoint_times(self, x, spaces):
        spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))
        if spaces is None:
            # this case means that x lives on only one space, which is
            # identical to the space in the domain of `self`. Otherwise the
            # input check of LinearOperator would have failed.
            axes = x.domain_axes[0]
        else:
            axes = x.domain_axes[spaces[0]]
        axis = axes[0]
        nval = len(self._dposc)
        prefix = (slice(None),) * axis
        sl_l = prefix + (slice(None, -1),)  # "left" slice
        sl_r = prefix + (slice(1, None),)  # "right" slice
        dpos = self._dpos.reshape((1,)*axis + (nval-1,))
        dposc = self._dposc.reshape((1,)*axis + (nval,))
        y = x.copy().weight(power=0.5).val
        y /= sqrt(dposc)
        y[prefix + (slice(None, 2),)] = 0.
        y[prefix + (-1,)] = 0.
        deriv = (y[sl_r]-y[sl_l])/dpos  # defined between points
        ret = np.empty_like(x.val)
        ret[sl_l] = deriv
        ret[prefix + (-1,)] = 0.
        ret[sl_r] -= deriv
        return Field(self.domain, val=ret).weight(-1, spaces=spaces)
