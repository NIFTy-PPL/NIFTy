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

from ..compat import *
from ..domain_tuple import DomainTuple
from ..field import Field
from .linear_operator import LinearOperator


class DomainTupleFieldInserter(LinearOperator):
    def __init__(self, domain, new_space, ind, infront=False):
        '''Writes the content of a field into one slice of a DomainTuple.

        Parameters
        ----------
        domain : Domain, tuple of Domain or DomainTuple
        new_space : Domain, tuple of Domain or DomainTuple
        ind : Integer
            Index of the same space as new_space
        infront : Boolean
            If true, the new domain is added in the beginning of the
            DomainTuple. Otherwise it is added at the end.
        '''
        # FIXME Add assertions
        self._domain = DomainTuple.make(domain)
        if infront:
            self._target = DomainTuple.make([new_space] + list(self.domain))
        else:
            self._target = DomainTuple.make(list(self.domain) + [new_space])
        self._infront = infront
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._ind = ind

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = np.zeros(self.target.shape, dtype=x.dtype)
            if self._infront:
                res[self._ind] = x.to_global_data()
            else:
                res[..., self._ind] = x.to_global_data()
            return Field.from_global_data(self.target, res)
        else:
            if self._infront:
                return Field.from_global_data(self.domain,
                                              x.to_global_data()[self._ind])
            else:
                return Field.from_global_data(
                    self.domain,
                    x.to_global_data()[..., self._ind])
