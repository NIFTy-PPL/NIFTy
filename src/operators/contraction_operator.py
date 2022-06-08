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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import partial

import numpy as np

from .. import utilities
from ..domain_tuple import DomainTuple
from ..field import Field
from .linear_operator import LinearOperator


class ContractionOperator(LinearOperator):
    """A :class:`LinearOperator` which sums up fields into the direction of
    subspaces.

    This Operator sums up a field which is defined on a :class:`DomainTuple`
    to a :class:`DomainTuple` which is a subset of the former.

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
    spaces : None, int or tuple of int
        The elements of "domain" which are contracted.
        If `None`, everything is contracted
    power : int, default=0
        If nonzero, the fields defined on self.domain are weighted with the
        specified power along the submdomains which are contracted.
    """

    def __init__(self, domain, spaces, power=0):
        self._domain = DomainTuple.make(domain)
        self._spaces = utilities.parse_spaces(spaces, len(self._domain))
        self._target = [
            dom for i, dom in enumerate(self._domain) if i not in self._spaces
        ]
        self._target = DomainTuple.make(self._target)
        self._power = power
        self._capability = self.TIMES | self.ADJOINT_TIMES

        try:
            from jax import numpy as jnp
            from jax.tree_util import tree_map

            from ..nifty2jax import spaces_to_axes

            fct = jnp.array(1.)
            wgt = jnp.array(1.)
            if self._power != 0:
                for ind in self._spaces:
                    wgt_spc = self._domain[ind].dvol
                    if np.isscalar(wgt_spc):
                        fct *= wgt_spc
                    else:
                        new_shape = np.ones(len(self._domain.shape), dtype=np.int64)
                        new_shape[self._domain.axes[ind][0]:
                                  self._domain.axes[ind][-1]+1] = wgt_spc.shape
                        wgt *= wgt_spc.reshape(new_shape)**power
                fct = fct**power

            def weighted_space_sum(x):
                if self._power != 0:
                    x = fct * wgt * x
                axes = spaces_to_axes(self._domain, self._spaces)
                return tree_map(partial(jnp.sum, axis=axes), x)

            self._jax_expr = weighted_space_sum
        except ImportError:
            self._jax_expr = None

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.ADJOINT_TIMES:
            ldat = x.val
            shp = []
            for i, dom in enumerate(self._domain):
                tmp = dom.shape
                shp += tmp if i not in self._spaces else (1,)*len(dom.shape)
            ldat = np.broadcast_to(ldat.reshape(shp), self._domain.shape)
            res = Field(self._domain, ldat)
            if self._power != 0:
                res = res.weight(self._power, spaces=self._spaces)
            return res
        else:
            if self._power != 0:
                x = x.weight(self._power, spaces=self._spaces)
            res = x.sum(self._spaces)
            return res if isinstance(res, Field) else Field.scalar(res)


def IntegrationOperator(domain, spaces):
    """A :class:`LinearOperator` which integrates fields into the direction
    of subspaces.

    This Operator integrates a field which is defined on a :class:`DomainTuple`
    to a :class:`DomainTuple` which is a subset of the former.

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
    spaces : None, int or tuple of int
        The elements of "domain" which are contracted.
        If `None`, everything is contracted
    """
    return ContractionOperator(domain, spaces, 1)
