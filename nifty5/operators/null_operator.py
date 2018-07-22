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
from ..domain_tuple import DomainTuple
from ..field import Field
from ..multi.multi_field import MultiField
from ..operators.linear_operator import LinearOperator


class NullOperator(LinearOperator):
    """Operator corresponding to a matrix of all zeros.

    Parameters
    ----------
    domain : DomainTuple or MultiDomain
        input domain
    target : DomainTuple or MultiDomain
        output domain
    """

    def __init__(self, domain, target):
        from ..sugar import makeDomain
        self._domain = makeDomain(domain)
        self._target = makeDomain(target)

    @staticmethod
    def _nullfield(dom):
        if isinstance(dom, DomainTuple):
            return Field.full(dom, 0)
        else:
            return MultiField(dom, (None,)*len(dom))

    def apply(self, x, mode):
        self._check_input(x, mode)

        if mode == self.TIMES:
            return self._nullfield(self._target)
        return self._nullfield(self._domain)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
