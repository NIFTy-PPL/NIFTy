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
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from .linear_operator import LinearOperator


class ValueInserter(LinearOperator):
    """Operator which inserts one value into a field.

    Parameters
    ----------
    target : Domain, tuple of Domain or DomainTuple
    index : tuple
        The index of the target into which the value of the domain shall be
        written.
    """

    def __init__(self, target, index):
        from ..sugar import makeDomain
        self._domain = makeDomain(UnstructuredDomain(1))
        self._target = DomainTuple.make(target)
        if not isinstance(index, tuple):
            raise TypeError
        self._index = index
        self._capability = self.TIMES | self.ADJOINT_TIMES
        # Check if index is in bounds
        np.empty(self.target.shape)[self._index]

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.to_global_data()
        if mode == self.TIMES:
            res = np.zeros(self.target.shape, dtype=x.dtype)
            res[self._index] = x
        else:
            res = np.full((1,),x[self._index])
        return Field.from_global_data(self._tgt(mode), res)
