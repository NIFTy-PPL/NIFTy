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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import reduce
from operator import mul

import numpy as np

from ..domain_tuple import DomainTuple
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..sugar import makeDomain
from .linear_operator import LinearOperator


class ValueInserter(LinearOperator):
    # FIXME THIS IS NOT A LINEAR OPERATOR
    """Inserts one value into a field which is constant otherwise.

    Parameters
    ----------
    target : Domain, tuple of Domain or DomainTuple
    index : iterable of int
        The index of the target into which the value of the domain shall be
        inserted.
    default_value : float
        Constant value which is inserted everywhere where the input operator
        is not inserted. Default is 0.
    """

    def __init__(self, target, index, default_value=0.):
        self._domain = makeDomain(UnstructuredDomain(1))
        self._target = DomainTuple.make(target)

        # Type and value checks
        index = tuple(index)
        if not all([
                isinstance(n, int) and n >= 0 and n < self.target.shape[i]
                for i, n in enumerate(index)
        ]):
            raise TypeError
        if not len(index) == len(self.target.shape):
            raise ValueError
        np.empty(self.target.shape)[index]

        self._index = index
        self._dv = float(default_value)
        self._dvsum = self._dv*(reduce(mul, self.target.shape) - 1)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.to_global_data()
        if mode == self.TIMES:
            res = np.full(self.target.shape, self._dv, dtype=x.dtype)
            res[self._index] = x
        else:
            res = np.full((1,), x[self._index] + self._dvsum, dtype=x.dtype)
        return Field.from_global_data(self._tgt(mode), res)
