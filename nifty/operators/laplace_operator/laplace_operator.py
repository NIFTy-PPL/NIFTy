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
from nifty.field import Field
from nifty.spaces.power_space import PowerSpace
from nifty.operators.endomorphic_operator import EndomorphicOperator

import nifty.nifty_utilities as utilities


class LaplaceOperator(EndomorphicOperator):
    """A irregular LaplaceOperator with free boundary and excluding monopole.

    This LaplaceOperator implements the second derivative of a Field in PowerSpace
    on logarithmic or linear scale with vanishing curvature at the boundary, starting
    at the second entry of the Field. The second derivative of the Field on the irregular grid
    is calculated using finite differences.

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

        self._dist_l = pos[1:-1]-pos[:-2]
        self._dist_r = pos[2:]-pos[1:-1]
        self._dist_c = 0.5*(pos[2:]-pos[:-2])

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
        nval = len(self._dist_l)
        prefix = (slice(None),) * axis
        sl_c = prefix + slice(1,-1)
        sl_l = prefix + slice(None,-2)
        sl_r = prefix + slice(2,None)
        dist_l = self._dist_l.reshape((1,)*axis + (nval,))
        dist_r = self._dist_r.reshape((1,)*axis + (nval,))
        dist_c = self._dist_c.reshape((1,)*axis + (nval,))
        dx_r = x[sl_r] - x[sl_c]
        dx_l = x[sl_c] - x[sl_l]
        ret = x.val.copy_empty()
        ret[sl_c] = (dx_r/dist_r - dx_l/dist_l)/sqrt(dist_c)
        ret[prefix + slice(None,2)] = 0.
        res[prefix + slice(-1,None)] = 0.
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
        nval = len(self._dist_l)
        prefix = (slice(None),) * axis
        sl_c = prefix + slice(1,-1)
        sl_l = prefix + slice(None,-2)
        sl_r = prefix + slice(2,None)
        dist_l = self._dist_l.reshape((1,)*axis + (nval,))
        dist_r = self._dist_r.reshape((1,)*axis + (nval,))
        dist_c = self._dist_c.reshape((1,)*axis + (nval,))
        dx_r = x[sl_r] - x[sl_c]
        dx_l = x[sl_c] - x[sl_l]
        y = x.copy().weight(power=0.5).val
        y[sl_c] *= sqrt(dist_c)
        y[prefix + slice(None, 2)] = 0.
        y[prefix + slice(-1, None)] = 0.
        ret = y.copy_empty()
        y[sl_c] /= dist_c
        ret[sl_c] = (y[sl_r]-y[sl_c])/dist_r - (y[sl_c]-y[sl_l])/dist_l

        ret[prefix + (0,)] = y[prefix+(1,)] / dist_l[prefix+(0,)]
        ret[prefix + (-1,)] = y[prefix+(-2,)] / dist_r[prefix + (-1,)]
        return Field(self.domain, val=ret).weight(-1, spaces=spaces)

Laplace:
L = (dxr/dr - dxl/dl)/  sqrt(dc)

adjoint Laplace:

tmp = x/sqrt(dc)
tmp2 = (tmpr-tmp)/dr - (tmp-tmpl)/dl
