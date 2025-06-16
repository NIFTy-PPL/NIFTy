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

    def apply(self, x, mode):
        self._check_input(x, mode)
        dev0 = x.device_id
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
            assert res.device_id == dev0
        else:
            if self._power != 0:
                x = x.weight(self._power, spaces=self._spaces)
            assert x.device_id == dev0
            res = x.sum(self._spaces)
            assert res.device_id == dev0
            if np.isscalar(res):
                res = Field.scalar(res).at(dev0, check_fail=False)
            assert res.device_id == dev0
        return res


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
