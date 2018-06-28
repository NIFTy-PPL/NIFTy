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

from ..domain_tuple import DomainTuple
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..sugar import full
from .linear_operator import LinearOperator


class MaskOperator(LinearOperator):
    def __init__(self, domain, target, xy):
        self._domain = DomainTuple.make(domain)
        # TODO Takes a field (boolean or 0/1)
        # TODO Add MultiFields (output MultiField of unstructured domains)

        assert len(xy.shape) == 2
        assert xy.shape[1] == 2
        self._target = UnstructuredDomain(xy.shape[0])

        self._xs = xy.T[0]
        self._ys = xy.T[1]

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = x.val[self._xs, self._ys]
            return Field(self.target, res)
        res = full(self.domain, 0.)
        res[self._xs, self._ys] = x.val
        return Field(self.domain, res)

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target
