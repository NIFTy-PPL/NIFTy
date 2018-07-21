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

from ..compat import *
from ..multi.multi_domain import MultiDomain
from ..field import Field
from .linear_operator import LinearOperator


class SelectionOperator(LinearOperator):
    """Extracts from a MultiField a copy of the Field
    living on the subdomain selected by `key`.

    Parameters
    ----------
    domain : :class:`MultiDomain`
        Domain of the MultiFields to be acted on
    key : :class:`str`
        String identifier of the wanted subdomain
    """

    def __init__(self, domain, key):
        self._domain = MultiDomain.make(domain)
        self._key = key

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._domain[self._key]

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            f = x[self._key]
            return Field.full(self.target, 0) if f is None else f
        else:
            from ..multi.multi_field import MultiField
            return MultiField.from_dict({self._key: x}, self._domain)
